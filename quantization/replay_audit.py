#!/usr/bin/env python3
"""Replay a spread of captured requests VERBATIM against running engines and
audit the responses: are they correct, and do they spend tokens they do not need to?

Requests are sent with every logged field (no allow-list), directly to the engines.
Privacy: prints verdicts, counts and field paths only — never request or response text.

Usage: python3 harness/replay_audit.py [--ports 9001,9002] [--per-shape 5]
"""
import argparse, collections, glob, hashlib, json, re, urllib.request, urllib.error
import os
from concurrent.futures import ThreadPoolExecutor

try:
    import jsonschema
except ImportError:
    jsonschema = None

LOGS = os.environ.get("TRAFFIC_LOGS", "traffic-logs/traffic-*.jsonl")
# our own probes, not app traffic
MARK = ("capital of", "Count one to", "city number", "billing service", "Hamlet", "19.50 euro",
        "tank fills", "pump adds", "quarterly review noted", "Porto", "Oslo", "Norway", "Lisbon",
        "Italy", "Portugal", "Set text to exactly", "ripe banana", "Vardas", "Say hi", "Say ok",
        'the word "hello"')
BAD_PREFIX = re.compile(r'^\s*[:,>;=\]\}]')


def thinking(q):
    ctk = q.get("chat_template_kwargs") or {}
    return not (q.get("reasoning_effort") == "none" or ctk.get("enable_thinking") is False)


def shape(r):
    q = r["request"]
    return ("tools" if q.get("tools") else "json" if q.get("response_format") else "plain",
            "think" if thinking(q) else "nothink", "stream" if r.get("streamed") else "nostream")


def load(per_shape):
    rows = []
    for f in sorted(glob.glob(LOGS)):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            q = r.get("request") or {}
            if r.get("status") == 200 and q.get("messages") and not any(m in json.dumps(q, ensure_ascii=False) for m in MARK):
                rows.append(r)
    by = collections.defaultdict(list)
    for r in rows:
        by[shape(r)].append(r)
    picked = []
    for sh, rs in sorted(by.items()):
        # spread: one per distinct (system prompt, schema/tools) first, then by prompt size
        seen, uniq = set(), []
        for r in rs:
            q = r["request"]
            k = hashlib.md5(json.dumps([(q["messages"][0].get("content") or "")[:400],
                                        q.get("response_format"), q.get("tools")], sort_keys=True,
                                       ensure_ascii=False).encode()).hexdigest()
            if k not in seen:
                seen.add(k); uniq.append(r)
        uniq.sort(key=lambda r: len(json.dumps(r["request"])))
        n = min(per_shape, len(uniq))
        picked += [(sh, uniq[round(i * (len(uniq) - 1) / max(n - 1, 1))]) for i in range(n)]
    return len(rows), picked


def post(port, path, body, timeout=900):
    q = urllib.request.Request(f"http://localhost:{port}{path}", json.dumps(body).encode(),
                               {"Content-Type": "application/json"})
    return urllib.request.urlopen(q, timeout=timeout)


def ntok(port, text):
    if not text:
        return 0
    return len(json.load(post(port, "/tokenize", {"model": "qwen", "prompt": text,
                                                   "add_special_tokens": False}))["tokens"])


def call(port, q, streamed):
    body = dict(q)
    if streamed:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    out = {"content": "", "reasoning": "", "tools": {}, "finish": None, "usage": None}
    if not streamed:
        d = json.load(post(port, "/v1/chat/completions", body))
        ch = d["choices"][0]; m = ch["message"]
        out.update(content=m.get("content") or "", reasoning=m.get("reasoning") or "",
                   finish=ch.get("finish_reason"), usage=d.get("usage"),
                   content_is_null=m.get("content") is None)
        out["tools"] = {i: {"name": t["function"]["name"], "args": t["function"]["arguments"]}
                        for i, t in enumerate(m.get("tool_calls") or [])}
        return out
    for line in post(port, "/v1/chat/completions", body):
        line = line.decode().strip()
        if not line.startswith("data: {"):
            continue
        d = json.loads(line[6:])
        if d.get("usage"):
            out["usage"] = d["usage"]
        for ch in d.get("choices") or []:
            de = ch.get("delta") or {}
            out["content"] += de.get("content") or ""
            out["reasoning"] += de.get("reasoning") or ""
            for t in de.get("tool_calls") or []:
                s = out["tools"].setdefault(t["index"], {"name": "", "args": ""})
                f = t.get("function") or {}
                s["name"] += f.get("name") or ""
                s["args"] += f.get("arguments") or ""
            out["finish"] = ch.get("finish_reason") or out["finish"]
    return out


def scan(obj, out, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            scan(v, out, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            scan(v, out, f"{path}[{i}]")
    elif isinstance(obj, str):
        if BAD_PREFIX.match(obj):
            out.append(f"{path}:leading-delim")
        elif obj != obj.strip():
            out.append(f"{path}:padded-string")


def repetition(text):
    """Share of 8-word windows that are repeats — degenerate loops show up here."""
    w = text.split()
    if len(w) < 40:
        return 0.0
    grams = [" ".join(w[i:i + 8]) for i in range(len(w) - 7)]
    return 1 - len(set(grams)) / len(grams)


def audit(port, q, o):
    """-> (problems, waste, numbers)."""
    prob, waste = [], []
    c, fin = o["content"], o["finish"]
    u = o["usage"] or {}
    comp = u.get("completion_tokens") or 0
    r_tok = ((u.get("completion_tokens_details") or {}).get("reasoning_tokens")
             or ntok(port, o["reasoning"]))
    c_tok = ntok(port, c)
    if fin == "length":
        prob.append("hit-max_tokens")
    if not thinking(q) and o["reasoning"]:
        prob.append("reasoning-with-thinking-off")
    if c != c.lstrip():
        waste.append("content-leading-ws")
    if c != c.rstrip():
        waste.append("content-trailing-ws")
    if "<think>" in c or "</think>" in c:
        prob.append("think-tag-in-content")
    if repetition(c) > 0.2 or repetition(o["reasoning"]) > 0.2:
        prob.append("repetition-loop")

    compact_tok = None
    rf = q.get("response_format") or {}
    if rf.get("type") == "json_schema" and fin != "length":
        try:
            obj = json.loads(c)
        except Exception:
            prob.append("json-unparseable"); obj = None
        if obj is not None:
            sch = (rf.get("json_schema") or {}).get("schema")
            if jsonschema and sch:
                errs = list(jsonschema.Draft202012Validator(sch).iter_errors(obj))
                if errs:
                    prob.append(f"schema-invalid:{'/'.join(map(str, errs[0].path))}")
            flags = []; scan(obj, flags)
            prob += flags[:3]
            compact_tok = ntok(port, json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            if c_tok > compact_tok * 1.15 + 3:
                waste.append(f"json-whitespace:+{c_tok - compact_tok}tok")

    if q.get("tools"):
        names = {t["function"]["name"]: t["function"].get("parameters") or {} for t in q["tools"]}
        if fin == "tool_calls" and not o["tools"]:
            prob.append("finish=tool_calls-but-none")
        for t in o["tools"].values():
            if t["name"] not in names:
                prob.append("unknown-tool"); continue
            try:
                args = json.loads(t["args"] or "{}")
            except Exception:
                prob.append("tool-args-unparseable"); continue
            miss = [k for k in names[t["name"]].get("required", []) if k not in args]
            if miss:
                prob.append(f"tool-args-missing:{len(miss)}")
            extra = [k for k in args if k not in (names[t["name"]].get("properties") or args)]
            if extra:
                prob.append(f"tool-args-unknown:{len(extra)}")
            flags = []; scan(args, flags); prob += flags[:2]
        if o["tools"] and c.strip():
            waste.append(f"prose-alongside-tool-call:{c_tok}tok")
    elif not c.strip() and fin != "length":
        prob.append("empty-content")

    budget = q.get("thinking_token_budget")
    if budget and r_tok > budget * 1.1 + 20:
        prob.append(f"thinking-budget-exceeded:{r_tok}>{budget}")
    return prob, waste, {"comp": comp, "reason": r_tok, "content": c_tok, "compact": compact_tok,
                         "prompt": u.get("prompt_tokens") or 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ports", default="9001,9002")
    ap.add_argument("--per-shape", type=int, default=5)
    a = ap.parse_args()
    ports = a.ports.split(",")
    total, picked = load(a.per_shape)
    print(f"{total} real logged requests; replaying {len(picked)} x {len(ports)} engines"
          f"{'' if jsonschema else '   (jsonschema not installed: schema validation skipped)'}\n")
    print(f"{'#':>2} {'shape':28s} {'port':4s} {'fin':10s} {'prompt':>7s} {'comp':>6s} {'reason':>6s} "
          f"{'content':>7s} {'logged':>6s}  verdict")
    tot = collections.Counter(); P = collections.Counter(); W = collections.Counter()
    sums = collections.defaultdict(lambda: collections.Counter())

    def one(args):
        i, sh, r, port = args
        try:
            o = call(port, r["request"], r.get("streamed"))
            return i, sh, r, port, o, audit(port, r["request"], o)
        except Exception as e:
            return i, sh, r, port, None, f"{type(e).__name__}: {str(e)[:80]}"

    jobs = [(i, sh, r, p) for i, (sh, r) in enumerate(picked) for p in ports]
    with ThreadPoolExecutor(max_workers=len(ports) * 2) as ex:
        for i, sh, r, port, o, res in ex.map(one, jobs):
            name = "/".join(sh)
            if o is None:
                print(f"{i:2d} {name:28s} {port:4s} ERROR {res}"); tot["error"] += 1; continue
            prob, waste, n = res
            logged = (r.get("usage") or {}).get("completion_tokens")
            verdict = "OK" if not prob and not waste else " ".join(["!" + p for p in prob] + ["~" + w for w in waste])
            print(f"{i:2d} {name:28s} {port:4s} {str(o['finish']):10s} {n['prompt']:7d} {n['comp']:6d} "
                  f"{n['reason']:6d} {n['content']:7d} {str(logged or '-'):>6s}  {verdict}", flush=True)
            tot["ok" if not prob else "problem"] += 1
            for p in prob: P[p.split(":")[0]] += 1
            for w in waste: W[w.split(":")[0]] += 1
            s = sums[name]
            s["n"] += 1; s["comp"] += n["comp"]; s["reason"] += n["reason"]; s["content"] += n["content"]
            if n["compact"] is not None:
                s["json_tok"] += n["content"]; s["compact"] += n["compact"]
            if logged:
                s["logged"] += logged; s["now_vs_logged"] += n["comp"]

    print(f"\nresponses: {dict(tot)}")
    print(f"problems (!): {dict(P) or 'none'}")
    print(f"waste    (~): {dict(W) or 'none'}")
    print(f"\n{'shape':28s} {'n':>3s} {'comp/req':>9s} {'reasoning%':>10s} {'json vs compact':>16s} {'now vs logged':>14s}")
    for name, s in sorted(sums.items()):
        jc = f"{s['json_tok']}/{s['compact']} (+{100*(s['json_tok']-s['compact'])/max(s['compact'],1):.0f}%)" if s["compact"] else "-"
        nl = f"{s['now_vs_logged']}/{s['logged']}" if s["logged"] else "-"
        print(f"{name:28s} {s['n']:3d} {s['comp']/s['n']:9.0f} {100*s['reason']/max(s['comp'],1):9.0f}% {jc:>16s} {nl:>14s}")


if __name__ == "__main__":
    main()
