"""Secondary teacher-forced logit analysis. Usage: python logit.py MODEL [--npairs N].
Resumable: per-pair records -> RESULTS/<model>_logit.jsonl (appended per batch, done batches skipped);
summary -> RESULTS/<model>_logit.json."""
import sys, json, random, time, os, argparse
import numpy as np, torch
from transformers import AutoTokenizer
import rp

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--npairs", type=int, default=200)
args = ap.parse_args()
name = args.name
hybrid = "3.5" in name
NPAIRS, BS = args.npairs, 16
BODIES = ["And everybody came to see the", "And then I looked again at the", "We all went out to find the"]
toks = [AutoTokenizer.from_pretrained(n) for n in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen2.5-0.5B-Instruct")]
LINES = rp.lines(toks)
m, tok = rp.load(name)
rng = random.Random(0); t0 = time.time()
os.makedirs(rp.RESULTS, exist_ok=True)
stem = os.path.join(rp.RESULTS, name.split("/")[-1])
print(f"{name} on {rp.DEV} {rp.DTYPE} kernels={rp.kernel_info()}", flush=True)
pairs = [(a, b) for a in LINES for b in LINES if a[0] == b[0] and a[2] != b[2]]
pairs = sorted(rng.sample(pairs, NPAIRS), key=lambda p: p[0][0])
WORDS = {c: [w for (_, w, cc) in LINES if cc == c] for c in rp.CLASSES}
TID = {w: tok(" " + w).input_ids[0] for ws in WORDS.values() for w in ws}
CONDS = ["clean_T", "clean_S", "NL", "NL_rand", "KV", "KV_rand"] + \
        (["STATE", "STATE_rand", "STATE_rec", "STATE_rec_rand", "BOTH"] if hybrid else [])

def ld(logp, t, s):
    """logsumexp over T-class minus S-class first-token logprobs, excluding the two line-1 words."""
    def lse(c, excl):
        ids = list({TID[w] for w in WORDS[c] if w not in excl})
        return torch.logsumexp(logp[ids], 0).item() if ids else float("nan")
    return lse(t[2], (t[1], s[1])) - lse(s[2], (t[1], s[1]))

@torch.no_grad()
def run(ids, n_mem, cache_fn=None, nl_patch=None):
    """Prefill first n_mem tokens (optionally patch residual at its last pos / patch cache), then the rest."""
    cache, _, _ = rp.prefill(m, ids[:, :n_mem], nl_patch=nl_patch)
    if cache_fn:
        cache_fn(cache)
    return m(input_ids=ids[:, n_mem:], past_key_values=cache, use_cache=True).logits[:, -1].float().log_softmax(-1)

jl = stem + "_logit.jsonl"
recs = [json.loads(x) for x in open(jl)] if os.path.exists(jl) else []
done = {r["batch"] for r in recs}
batches = [g[i:i + BS] for ti in range(len(rp.TEMPLATES)) for g in [[p for p in pairs if p[0][0] == ti]] for i in range(0, len(g), BS)]
assert all(len([r for r in recs if r["batch"] == b]) == len(batches[b]) * len(BODIES) for b in done), "corrupt jsonl; delete it"
for bi, sub in enumerate(batches):
    if bi in done:
        continue
    gen = torch.Generator().manual_seed(1000 + bi)  # per-batch seed: rand controls independent of resume point
    new = []
    for body in BODIES:
        idT = rp.encode(tok, [rp.prompt(*t[:2]) + body for t, _ in sub])
        idS = rp.encode(tok, [rp.prompt(*s[:2]) + body for _, s in sub])
        nP = len(tok(rp.prompt(*sub[0][0][:2])).input_ids)      # prompt length; newline at nP-1
        _, _, capT = rp.prefill(m, idT[:, :nP], capture=True)
        _, _, capS = rp.prefill(m, idS[:, :nP], capture=True)
        cS, _, _ = rp.prefill(m, idS[:, :nP - 1])
        out = {"clean_T": run(idT, nP - 1), "clean_S": run(idS, nP - 1)}
        out["NL"] = run(idT, nP, nl_patch=capS)
        out["NL_rand"] = run(idT, nP, nl_patch=[rp.rand_like_delta(a, b, gen) for a, b in zip(capT, capS)])
        for c in CONDS[4:]:
            kinds = ["state", "kv"] if c == "BOTH" else [c.split("_rand")[0].lower()]
            out[c] = run(idT, nP - 1, cache_fn=lambda ca: [rp.patch_cache(ca, cS, k, rand=c.endswith("_rand"), gen=gen) for k in kinds])
        for j, (t, s) in enumerate(sub):
            new.append(dict(batch=bi, T=t[1], S=s[1], body=body, **{c: ld(out[c][j], t, s) for c in CONDS}))
    with open(jl, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in new)); f.flush(); os.fsync(f.fileno())
    recs += new
    print(f"batch {bi + 1}/{len(batches)} {time.time()-t0:.0f}s", flush=True)

# fraction restored, per (pair, body); drop rows with no clean rhyme signal; average bodies per pair
rows = {}
for r in recs:
    gap = r["clean_T"] - r["clean_S"]
    if np.isnan(gap) or abs(gap) < 0.5:
        continue
    rows.setdefault((r["T"], r["S"]), []).append({c: (r["clean_T"] - r[c]) / gap for c in CONDS[2:]})
F = {c: np.array([np.mean([x[c] for x in v]) for v in rows.values()]) for c in CONDS[2:]}
b = np.random.default_rng(0)
summ = {}
for c, v in F.items():
    if not len(v):
        summ[c] = [float("nan")] * 4; continue
    bs = [b.choice(v, len(v)).mean() for _ in range(2000)]
    summ[c] = [round(float(v.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4), round(float(np.median(v)), 4)]
gaps = np.array([r["clean_T"] - r["clean_S"] for r in recs])
print(f"clean gap LD_T-LD_S: mean {np.nanmean(gaps):.3f}, frac>0 {np.nanmean(gaps > 0):.3f}, rows kept {sum(map(len, rows.values()))}/{len(recs)}, pairs {len(rows)}")
for c, v in summ.items():
    print(f"{c:15s} f mean {v[0]} CI [{v[1]}, {v[2]}] median {v[3]}")
json.dump(dict(model=name, device=rp.DEV, dtype=str(rp.DTYPE), kernels=rp.kernel_info(), summary=summ,
               gap_mean=float(np.nanmean(gaps)), gap_pos=float(np.nanmean(gaps > 0)),
               n_pairs_kept=len(rows), records=recs, seconds=time.time() - t0),
          open(stem + "_logit.json", "w"), indent=1)
