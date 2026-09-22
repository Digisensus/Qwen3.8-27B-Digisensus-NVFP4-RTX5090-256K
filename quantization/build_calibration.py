#!/usr/bin/env python3
"""Build the calibration set: token ids rendered through the model's OWN chat template,
so tool calls come out as Qwen3.x XML, reasoning as <think> blocks, JSON as the model
emits it. Calibration must look like what the model produces when serving.

Sources (workload-weighted; code is a sacrifice axis and is deliberately absent):
  traffic   real conversations + the model's real responses (request logs),
            only rows before --traffic-cutoff so later traffic stays clean for frozen-v2
  tools     NousResearch/hermes-function-calling-v1 single-turn, converted to OpenAI
            tools/tool_calls so the template renders the native format
  json      hermes json-mode single-turn + agentic
  prose     HuggingFaceH4/ultrachat_200k test_sft

Long sequences contribute a head window AND a tail window: the head carries the system
prompt / instructions, the tail carries the assistant response.

PRIVACY: with traffic enabled the output contains real user content. It is written under
data/ which must never be committed or published.

Usage: venv/bin/python harness/build_calibration.py --out data/calib-r0.jsonl [--no-traffic]
"""
import argparse, collections, glob, hashlib, json, random, re, sys
import os
import pandas as pd
from transformers import AutoTokenizer

BF16 = os.environ.get("BF16_MODEL", "models/Qwen3.8-27B")
SRC = "data/calib-src"
LOGS = os.environ.get("TRAFFIC_LOGS", "traffic-logs/traffic-*.jsonl")
MARK = ("capital of", "Count one to", "city number", "billing service", "Hamlet", "19.50 euro",
        "tank fills", "pump adds", "quarterly review noted", "Porto", "Oslo", "Norway", "Lisbon",
        "Italy", "Portugal", "Set text to exactly", "ripe banana", "Vardas", "Say hi", "Say ok",
        'the word "hello"')


def thinking(q):
    ctk = q.get("chat_template_kwargs") or {}
    return not (q.get("reasoning_effort") == "none" or ctk.get("enable_thinking") is False)


def norm_tool_calls(tcs):
    out = []
    for t in tcs or []:
        f = t.get("function") or {}
        args = f.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        out.append({"type": "function", "function": {"name": f.get("name"), "arguments": args or {}}})
    return out


def norm_messages(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = norm_tool_calls(m["tool_calls"])
        if isinstance(m.get("content"), list):   # OpenAI content parts -> text
            m["content"] = "".join(p.get("text", "") for p in m["content"] if isinstance(p, dict))
        out.append(m)
    return out


def from_traffic(cutoff):
    seen, rows = set(), []
    for f in sorted(glob.glob(LOGS)):
        for l in open(f):
            try:
                r = json.loads(l)
            except Exception:
                continue
            q, resp = r.get("request") or {}, r.get("response") or {}
            if (r.get("status") != 200 or not q.get("messages") or r.get("ts", "") >= cutoff
                    or resp.get("finish_reason") not in ("stop", "tool_calls")
                    or any(m in json.dumps(q, ensure_ascii=False) for m in MARK)):
                continue
            h = hashlib.md5(json.dumps([q["messages"], resp], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            asst = {"role": "assistant", "content": resp.get("content") or ""}
            if resp.get("reasoning") and thinking(q):
                asst["reasoning_content"] = resp["reasoning"]
            if resp.get("tool_calls"):
                asst["tool_calls"] = resp["tool_calls"]
            rows.append({"src": "traffic", "messages": norm_messages(q["messages"] + [asst]),
                         "tools": q.get("tools"), "think": thinking(q)})
    return rows


HERMES_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def from_hermes_tools(n, rng):
    data = json.load(open(f"{SRC}/func-calling-singleturn.json"))
    rng.shuffle(data)
    rows = []
    for d in data:
        conv = d["conversations"]
        try:
            tools = json.loads(d["tools"]) if isinstance(d["tools"], str) else d["tools"]
        except Exception:
            continue
        user = next((c["value"] for c in conv if c["from"] == "human"), None)
        gpt = next((c["value"] for c in conv if c["from"] == "gpt"), None)
        if not tools or not user or not gpt:
            continue
        calls = []
        for m in HERMES_CALL.finditer(gpt):
            try:
                c = json.loads(m.group(1).replace("'", '"'))
                calls.append({"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments", {})}})
            except Exception:
                pass
        if not calls:
            continue
        rows.append({"src": "tools", "tools": tools, "think": False, "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": "", "tool_calls": calls}]})
        if len(rows) >= n:
            break
    return rows


def from_hermes_json(n, rng):
    data = json.load(open(f"{SRC}/json-mode-singleturn.json")) + json.load(open(f"{SRC}/json-mode-agentic.json"))
    rng.shuffle(data)
    rows = []
    role = {"system": "system", "human": "user", "gpt": "assistant"}
    for d in data[:n]:
        msgs = [{"role": role[c["from"]], "content": c["value"]} for c in d["conversations"] if c["from"] in role]
        rows.append({"src": "json", "tools": None, "think": False, "messages": msgs})
    return rows


def from_ultrachat(n, rng):
    df = pd.read_parquet(glob.glob(f"{SRC}/data/test_sft-*.parquet")[0])
    idx = rng.sample(range(len(df)), n)
    return [{"src": "prose", "tools": None, "think": False,
             "messages": [{"role": m["role"], "content": m["content"]} for m in df.iloc[i]["messages"]]}
            for i in idx]


def render(tok, row):
    kw = {"enable_thinking": bool(row["think"])}
    text = tok.apply_chat_template(row["messages"], tools=row["tools"] or None, tokenize=False,
                                   add_generation_prompt=False, **kw)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--traffic-cutoff", default="2026-09-21T00:00:00")
    ap.add_argument("--no-traffic", action="store_true")
    ap.add_argument("--n-tools", type=int, default=240)
    ap.add_argument("--n-json", type=int, default=240)
    ap.add_argument("--n-prose", type=int, default=160)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--target-tokens", type=int, default=640_000,
                    help="total budget; the cached activations cost tokens x 5120 x 2 B per layer boundary")
    ap.add_argument("--mix", default="traffic:0.55,tools:0.17,prose:0.17,json:0.11")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    tok = AutoTokenizer.from_pretrained(BF16)

    rows = [] if a.no_traffic else from_traffic(a.traffic_cutoff)
    rows += from_hermes_tools(a.n_tools, rng) + from_hermes_json(a.n_json, rng) + from_ultrachat(a.n_prose, rng)

    samples, fails = [], collections.Counter()
    for row in rows:
        try:
            ids = render(tok, row)
        except Exception as e:
            fails[f"{row['src']}:{type(e).__name__}"] += 1
            continue
        if len(ids) <= a.max_len:
            wins = [ids]
        else:   # head = instructions, tail = the response
            wins = [ids[:a.max_len], ids[-a.max_len:]]
        for w in wins:
            samples.append({"src": row["src"], "input_ids": w})
    # per-source token quotas
    mix = {k: float(v) for k, v in (x.split(":") for x in a.mix.split(","))}
    if a.no_traffic:
        mix.pop("traffic", None)
    z = sum(mix.values())
    pools = collections.defaultdict(list)
    for s in samples:
        pools[s["src"]].append(s)
    kept = []
    for src, share in mix.items():
        pool = pools[src]; rng.shuffle(pool); budget = a.target_tokens * share / z; used = 0
        for s in pool:
            if used >= budget:
                break
            kept.append(s); used += len(s["input_ids"])
        if used < budget * 0.9:
            print(f"  note: {src} has only {used:,} of its {budget:,.0f}-token quota")
    samples = kept
    rng.shuffle(samples)
    with open(a.out, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    by = collections.defaultdict(lambda: [0, 0])
    for s in samples:
        by[s["src"]][0] += 1; by[s["src"]][1] += len(s["input_ids"])
    tot = sum(v[1] for v in by.values())
    print(f"{len(samples)} samples, {tot:,} tokens -> {a.out}")
    for k, (n, t) in sorted(by.items()):
        print(f"  {k:8s} {n:4d} samples {t:9,d} tokens ({100 * t / tot:4.1f}%)")
    if fails:
        print("render failures:", dict(fails))
    # show the native format made it through, without printing content
    tr = [s for s in samples if s["src"] in ("tools", "traffic")]
    txt = [tok.decode(s["input_ids"]) for s in tr[:200]]
    print("markers: <tool_call>", sum("<tool_call>" in t for t in txt), " <function=", sum("<function=" in t for t in txt),
          " <think>", sum("<think>" in t for t in txt), f"(of {len(txt)} tools/traffic samples checked)")


if __name__ == "__main__":
    main()
