#!/usr/bin/env python3
"""End-to-end sensitivity scan: what does each quantization unit cost in output KL?

For every decision unit (a vLLM fused group in one layer — the granularity at which a
format can actually be chosen) and every candidate format, quantize ONLY that unit
(fake-quant: weights round-tripped through the format, activations quantized on the fly
by a pre-hook), run the eval set, measure KL of the output distribution against bf16,
restore. Result: cost[unit][format], the input to a bit-reallocation (knapsack).

Units (vLLM fuses these, so they must share a format):
  gdn.qkvz  = linear_attn.in_proj_qkv + in_proj_z       gdn.out  = linear_attn.out_proj
  attn.qkv  = self_attn.q/k/v_proj                        attn.o   = self_attn.o_proj
  mlp.gate_up = mlp.gate_proj + up_proj                   mlp.down = mlp.down_proj
  lm_head, embed (weight-only)
GDN in_proj_a/in_proj_b stay bf16 (tiny, the reference quant ignores them too).

Formats (fake-quant, round-to-nearest — GPTQ lowers every error, the RANKING is what matters):
  fp8   : W8A8 FP8 e4m3, per-output-channel weight scale, dynamic per-token activations
  nvfp4 : W4A4 NVFP4, e2m1 values, 16-wide groups, e4m3 group scales, fp32 global scale

Model is bf16, split over both GPUs. Results are appended per unit (resumable).
Usage: venv/bin/python harness/sensitivity.py --out results/sensitivity.jsonl
"""
import argparse, json, math, os, re, sys, time
import torch

BF16 = os.environ.get("BF16_MODEL", "models/Qwen3.8-27B")
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
F8MAX = 448.0


# ------------------------------------------------------------------ fake quantizers
def _e2m1(x):
    """Round |x| (already scaled so max is 6) to the nearest e2m1 value, keep sign."""
    grid = E2M1.to(x.device, torch.float32)
    mids = (grid[1:] + grid[:-1]) / 2
    a = x.abs().clamp(max=6.0)
    return torch.sign(x) * grid[torch.bucketize(a, mids)]


def fq_fp8_rows(x):
    """FP8 e4m3 with one scale per row (weight: per output channel; act: per token)."""
    xf = x.float()
    s = (xf.abs().amax(dim=-1, keepdim=True) / F8MAX).clamp(min=1e-12)
    return ((xf / s).to(torch.float8_e4m3fn).float() * s).to(x.dtype)


def fq_nvfp4(x):
    """NVFP4 along the last dim: 16-wide groups, e4m3 group scale, fp32 global scale."""
    shape = x.shape
    xf = x.float().reshape(-1, shape[-1])
    g = xf.reshape(xf.shape[0], -1, 16)
    gs = (F8MAX * 6.0 / xf.abs().amax().clamp(min=1e-12))           # global scale
    bs = (g.abs().amax(dim=-1, keepdim=True) / 6.0 * gs)              # group scale, pre-cast
    bs = bs.clamp(max=F8MAX).to(torch.float8_e4m3fn).float()
    eff = (bs / gs)
    q = torch.where(eff > 0, _e2m1(g / eff.clamp(min=1e-30)) * eff, torch.zeros_like(g))
    return q.reshape(shape).to(x.dtype)


FORMATS = {
    "fp8": (fq_fp8_rows, fq_fp8_rows),
    "nvfp4": (fq_nvfp4, fq_nvfp4),
    "fp8w": (fq_fp8_rows, None),          # weight-only (embeddings / lm_head option)
}


# ------------------------------------------------------------------ units
def units(model):
    lm = model.model.language_model
    out = []
    for i, layer in enumerate(lm.layers):
        if hasattr(layer, "linear_attn"):
            la = layer.linear_attn
            out.append((f"L{i:02d}.gdn.qkvz", [la.in_proj_qkv, la.in_proj_z]))
            out.append((f"L{i:02d}.gdn.out", [la.out_proj]))
        else:
            sa = layer.self_attn
            out.append((f"L{i:02d}.attn.qkv", [sa.q_proj, sa.k_proj, sa.v_proj]))
            out.append((f"L{i:02d}.attn.o", [sa.o_proj]))
        out.append((f"L{i:02d}.mlp.gate_up", [layer.mlp.gate_proj, layer.mlp.up_proj]))
        out.append((f"L{i:02d}.mlp.down", [layer.mlp.down_proj]))
    out.append(("lm_head", [model.lm_head]))
    out.append(("embed", [lm.embed_tokens]))
    return out


def band_units(model, size):
    """Same unit types, grouped over `size` consecutive layers. A single FP8 unit is below
    the bf16 numerical-noise floor (~4e-4 KL for ANY perturbation); a band carries 1-4 units
    of signal. Also returns, per band, the modules used for the noise-floor measurement."""
    per = {}
    for name, mods in units(model):
        if not name.startswith("L"):
            per[name] = (mods, [])
            continue
        layer, kind = int(name[1:3]), name.split(".", 1)[1]
        key = f"B{layer // size:02d}.{kind}"
        m, ls = per.setdefault(key, ([], []))
        m.extend(mods); ls.append(layer)
    return per


def chunked(fq, w, rows=16384):
    """Apply a row-wise fake quantizer in row chunks (lm_head/embed are 248k x 5120)."""
    if w.shape[0] <= rows:
        return fq(w)
    return torch.cat([fq(w[i:i + rows]) for i in range(0, w.shape[0], rows)])


def quantize_inplace(w, fq, rows=16384):
    """Fake-quantize weight matrix w IN PLACE, row chunk by row chunk. For NVFP4 the global
    scale is per tensor: a pinned extra row carries the full-tensor amax into every chunk."""
    amax = w.abs().amax().float() if fq is fq_nvfp4 else None
    for i in range(0, w.shape[0], rows):
        c = w[i:i + rows]
        if amax is not None:
            pad = c.new_zeros(1, c.shape[1]); pad[0, 0] = amax
            c.copy_(fq(torch.cat([c, pad]))[:-1])
        else:
            c.copy_(fq(c))


class Perturb:
    """Context manager: fake-quantize a unit in place, restore on exit."""
    def __init__(self, mods, fmt):
        self.mods, (self.wq, self.aq) = mods, FORMATS[fmt]
        self.saved, self.hooks = [], []

    def __enter__(self):
        for m in self.mods:
            big = m.weight.numel() > 2e8          # lm_head / embed: back up to CPU
            self.saved.append(m.weight.data.to("cpu") if big else m.weight.data.clone())
            quantize_inplace(m.weight.data, self.wq)
            if self.aq is not None and isinstance(m, torch.nn.Linear):
                aq = self.aq
                self.hooks.append(m.register_forward_pre_hook(lambda mod, inp: (aq(inp[0]),) + tuple(inp[1:])))
        return self

    def __exit__(self, *exc):
        for m, w in zip(self.mods, self.saved):
            m.weight.data.copy_(w)
        for h in self.hooks:
            h.remove()
        self.saved.clear()


# ------------------------------------------------------------------ evaluation
# Scored positions are ONLY model-written text (bf16 continuations, assistant turns).
# 2026-09-21: scoring on user-written ASR transcripts was noise-dominated — some positions
# are knife-edge, where even fla-vs-torch kernel rounding moves KL by 0.17 and 0.1%
# random weight noise costs as much as FP8. The model's own outputs are stable.
TOPK = 64
PAD = 0


@torch.no_grad()
def batch_chunks(model, batch, chunk=512):
    """batch: list of (ids, pos) — pos = positions whose NEXT token is scored.
    Right-padded (causal: pads never influence earlier positions). Yields
    (batch_index, offset, logprobs[chunk, vocab]) — callers reduce each chunk at once,
    full-vocab log-probs for a whole batch do not fit next to the weights."""
    L = max(len(ids) for ids, _ in batch)
    x = torch.full((len(batch), L), PAD, dtype=torch.long)
    for i, (ids, _) in enumerate(batch):
        x[i, :len(ids)] = ids
    dev = model.model.language_model.embed_tokens.weight.device
    h = model.model(input_ids=x.to(dev), use_cache=False).last_hidden_state
    for i, (_, pos) in enumerate(batch):
        hs = h[i, pos.to(h.device)].to(model.lm_head.weight.device)
        for off in range(0, len(hs), chunk):
            yield i, off, torch.log_softmax(model.lm_head(hs[off:off + chunk]).float(), -1)


def batches(samples, size):
    order = sorted(range(len(samples)), key=lambda i: len(samples[i][0]))
    for k in range(0, len(order), size):
        yield [order[j] for j in range(k, min(k + size, len(order)))]


@torch.no_grad()
def reference(model, samples, bs):
    parts = [[] for _ in samples]
    for idx in batches(samples, bs):
        for bi, off, lp in batch_chunks(model, [samples[j] for j in idx]):
            v, t = lp.topk(TOPK, dim=-1); parts[idx[bi]].append((v.cpu(), t.cpu()))
    return [(torch.cat([v for v, _ in p]), torch.cat([t for _, t in p])) for p in parts]


@torch.no_grad()
def kl_vs_ref(model, samples, ref, bs, tags, mask=None, want_pos=False):
    """KL(bf16 || perturbed) on the bf16 top-K support + tail bucket, averaged over scored
    positions. Returns (mean_kl, {tag: mean_kl}, top1 agreement)."""
    tot, n, agree = 0.0, 0, 0
    per_tag, posk = {}, {}
    for idx in batches(samples, bs):
        for bi, off, lp in batch_chunks(model, [samples[j] for j in idx]):
            i = idx[bi]
            rv, ri = (t[off:off + len(lp)].to(lp.device) for t in ref[i])
            q = lp.gather(-1, ri); p = rv.exp()
            kl = (p * (rv - q)).sum(-1)
            pt = (1 - p.sum(-1)).clamp(min=1e-12); qt = (1 - q.exp().sum(-1)).clamp(min=1e-12)
            kl = (kl + pt * (pt.log() - qt.log())).clamp(min=0)
            if want_pos:
                posk.setdefault(i, []).append(kl.cpu())
            if mask is not None:
                keep = mask[i][off:off + len(lp)].to(kl.device)
                kl, lpa, ria = kl[keep], lp.argmax(-1)[keep], ri[keep, 0]
            else:
                lpa, ria = lp.argmax(-1), ri[:, 0]
            agree += (lpa == ria).sum().item()
            tot += kl.sum().item(); n += kl.numel()
            t = per_tag.setdefault(tags[i], [0.0, 0]); t[0] += kl.sum().item(); t[1] += kl.numel()
    res = (tot / n, {k: v[0] / v[1] for k, v in per_tag.items()}, agree / n)
    if want_pos:
        return res + ({i: torch.cat(v) for i, v in posk.items()},)
    return res


def noisy(mods, rel, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    saved = [m.weight.data.clone() for m in mods]
    for m in mods:
        n = torch.randn(m.weight.shape, generator=g).to(m.weight.device, m.weight.dtype)
        m.weight.data.add_(n * m.weight.data.abs() * rel)
    return saved


def eval_set(ref_file, calib_file, per_tag, n_traffic, seq, tok):
    """Even-indexed frozen items (odd ones stay held out for the final evaluation):
    continuation positions only. Plus real traffic windows: assistant-turn positions only."""
    samples, tags = [], []
    counts = {}
    for k, l in enumerate(open(ref_file)):
        if k % 2:
            continue
        r = json.loads(l)
        if r["tag"] == "code" or counts.get(r["tag"], 0) >= per_tag:
            continue
        ids = torch.tensor(r["ids"][:seq])
        pos = torch.arange(max(r["n_ctx"] - 1, 0), len(ids) - 1)
        if len(pos) < 8:
            continue
        samples.append((ids, pos)); tags.append(r["tag"]); counts[r["tag"]] = counts.get(r["tag"], 0) + 1
    hdr = tok.encode("<|im_start|>assistant", add_special_tokens=False)
    got = 0
    for l in open(calib_file):
        r = json.loads(l)
        if r["src"] != "traffic" or got >= n_traffic:
            continue
        ids = r["input_ids"][:seq]
        starts = [i + len(hdr) for i in range(len(ids) - len(hdr)) if ids[i:i + len(hdr)] == hdr]
        if not starts or len(ids) - 1 - starts[-1] < 32:
            continue
        samples.append((torch.tensor(ids), torch.arange(starts[-1], len(ids) - 1)))
        tags.append("traffic_reply"); got += 1
    return samples, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=BF16)
    ap.add_argument("--calib", default="data/calib-r0.jsonl")
    ap.add_argument("--ref", default="results/reference-bf16.jsonl")
    ap.add_argument("--per-tag", type=int, default=10)
    ap.add_argument("--traffic", type=int, default=10)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--group", type=int, default=4, help="layers per band (1 = per-unit scan)")
    ap.add_argument("--unstable-kl", type=float, default=0.01, help="drop positions whose KL under 0.1%% noise reaches this")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--formats", default="nvfp4,fp8")
    ap.add_argument("--only", default="", help="regex on unit names (testing)")
    ap.add_argument("--split", type=int, default=32, help="last layer on GPU0")
    ap.add_argument("--cpu", action="store_true", help="CPU smoke test")
    a = ap.parse_args()
    t0 = time.time()
    from transformers import Qwen3_5ForConditionalGeneration

    kw = {"dtype": torch.bfloat16}
    if not a.cpu:
        # explicit split: vision tower is never used here (CPU); GPU1 also carries lm_head
        # and the full-vocab logit chunks, so it gets fewer layers
        dm = {"model.visual": "cpu", "model.language_model.embed_tokens": 0,
              "model.language_model.norm": 1, "model.language_model.rotary_emb": 1, "lm_head": 1}
        dm.update({f"model.language_model.layers.{i}": (0 if i <= a.split else 1) for i in range(64)})
        kw.update(device_map=dm)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, **kw).eval()
    print(f"[{time.time()-t0:5.0f}s] loaded; devices {sorted({str(p.device) for p in model.parameters()})}", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    samples, tags = eval_set(a.ref, a.calib, a.per_tag, a.traffic, a.seq, tok)
    print(f"eval: {len(samples)} sequences, {sum(len(p) for _, p in samples):,} scored positions "
          f"({sum(len(i) for i, _ in samples):,} tokens), {dict((t, tags.count(t)) for t in sorted(set(tags)))}", flush=True)
    t1 = time.time(); ref = reference(model, samples, a.batch)
    print(f"[{time.time()-t0:5.0f}s] reference in {time.time()-t1:.1f}s", flush=True)
    base, _, _ = kl_vs_ref(model, samples, ref, a.batch, tags)
    print(f"sanity: unperturbed KL vs reference = {base:.2e} (must be ~0)", flush=True)
    # Knife-edge positions flip under ANY perturbation (0.1% weight noise already costs
    # about as much KL as FP8 there). They add the same floor to every unit and bury the
    # differences between units. Find them with random-noise probes in 4 layers, drop them.
    U = dict(units(model))
    worst = None
    for k, u in enumerate(["L00.gdn.out", "L20.mlp.down", "L42.gdn.qkvz", "L62.gdn.out"]):
        saved = noisy(U[u], 1e-3, k)
        *_, pk = kl_vs_ref(model, samples, ref, a.batch, tags, want_pos=True)
        for m, w in zip(U[u], saved): m.weight.data.copy_(w)
        worst = pk if worst is None else {i: torch.maximum(worst[i], pk[i]) for i in pk}
    mask = {i: (worst[i] < a.unstable_kl) for i in worst}
    kept = sum(int(m.sum()) for m in mask.values()); total = sum(len(m) for m in mask.values())
    print(f"unstable positions (noise KL >= {a.unstable_kl}): {total - kept} of {total} dropped "
          f"({100 * (total - kept) / total:.1f}%)", flush=True)
    for u in ("L10.mlp.gate_up", "L50.mlp.down"):        # held-out units: floor vs signal
        saved = noisy(U[u], 1e-3, 99)
        fl, _, _ = kl_vs_ref(model, samples, ref, a.batch, tags, mask)
        for m, w in zip(U[u], saved): m.weight.data.copy_(w)
        vals = []
        for f in ("fp8", "nvfp4"):
            with Perturb(U[u], f):
                vals.append(kl_vs_ref(model, samples, ref, a.batch, tags, mask)[0])
        print(f"  floor check {u:16s} noise {fl:.2e}  fp8 {vals[0]:.2e} ({vals[0]/max(fl,1e-12):.1f}x)  "
              f"nvfp4 {vals[1]:.2e} ({vals[1]/max(fl,1e-12):.1f}x)", flush=True)

    done = set()
    if os.path.exists(a.out):
        for l in open(a.out):
            r = json.loads(l); done.add((r["unit"], r["fmt"]))
    groups = band_units(model, a.group)
    fmts = a.formats.split(",")
    todo = []
    for u, (mods, layers) in groups.items():
        if a.only and not re.search(a.only, u):
            continue
        for f in (["fp8w"] if u == "embed" else fmts):
            if (u, f) not in done:
                todo.append((u, mods, layers, f))
    bands = sorted({u.split(".")[0] for u in groups if u.startswith("B")})
    for b in bands:                     # noise floor per band (0.1% noise on its MLP down)
        if (f"{b}.floor", "noise") not in done and (not a.only or re.search(a.only, b)):
            todo.append((f"{b}.floor", groups[f"{b}.mlp.down"][0], groups[f"{b}.mlp.down"][1], "noise"))
    print(f"{len(todo)} measurements to do ({len(groups)} groups of {a.group} layers)", flush=True)
    with open(a.out, "a") as f:
        for k, (u, mods, layers, fmt) in enumerate(todo):
            t = time.time()
            if fmt == "noise":
                saved = noisy(mods, 1e-3, 1234)
                kl, per, top1 = kl_vs_ref(model, samples, ref, a.batch, tags, mask)
                for m, w in zip(mods, saved): m.weight.data.copy_(w)
            else:
                with Perturb(mods, fmt):
                    kl, per, top1 = kl_vs_ref(model, samples, ref, a.batch, tags, mask)
            params = sum(m.weight.numel() for m in mods)
            f.write(json.dumps({"unit": u, "fmt": fmt, "kl": kl, "top1": top1, "params": params,
                                "layers": layers, "per_tag": per}) + "\n"); f.flush()
            print(f"[{k+1:4d}/{len(todo)}] {u:18s} {fmt:6s} KL {kl:.3e} top1 {100*top1:6.2f}%  {time.time()-t:5.1f}s", flush=True)
    print(f"[{time.time()-t0:5.0f}s] done", flush=True)


if __name__ == "__main__":
    main()
