"""Primary experiment (free generation). Usage: python run.py MODEL [--pilot] [--nlines N --npairs N --any-pairs]
Resumable: clean stage -> RESULTS/<model>_clean.json(+ _resid.npy); pairs -> RESULTS/<model>.pairs.jsonl (one line per
pair, appended per batch, done batches skipped on restart); summary -> RESULTS/<model>.json.
--nlines/--npairs/--any-pairs are for smoke tests only (any-pairs ignores eligibility)."""
import sys, json, random, time, os, argparse
import numpy as np, torch
from transformers import AutoTokenizer
import rp

ap = argparse.ArgumentParser()
ap.add_argument("name"); ap.add_argument("--pilot", action="store_true")
ap.add_argument("--nlines", type=int, default=0); ap.add_argument("--npairs", type=int, default=200)
ap.add_argument("--any-pairs", action="store_true")
args = ap.parse_args()
name, pilot = args.name, args.pilot
NPAIRS, BS = args.npairs, 16
hybrid = "3.5" in name
toks = [AutoTokenizer.from_pretrained(n) for n in ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen2.5-0.5B-Instruct")]
LINES = rp.lines(toks)
m, tok = rp.load(name)
rng = random.Random(0)
t0 = time.time()
os.makedirs(rp.RESULTS, exist_ok=True)
stem = os.path.join(rp.RESULTS, name.split("/")[-1])
print(f"{name} on {rp.DEV} {rp.DTYPE} kernels={rp.kernel_info()}", flush=True)

def wilson(k, n, z=1.96):
    if n == 0:
        return [0, 0, 0]
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5 / d
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]

# ---- clean generations (+ newline residuals for the probe); saved once, reused on restart
lines = rng.sample(LINES, 60) if pilot else rng.sample(LINES, args.nlines) if args.nlines else LINES
if not pilot and os.path.exists(stem + "_clean.json") and os.path.exists(stem + "_resid.npy"):
    clean = {tuple(json.loads(k)): v for k, v in json.load(open(stem + "_clean.json")).items()}
    R = np.load(stem + "_resid.npy")
    print(f"loaded clean stage ({len(clean)} lines)", flush=True)
else:
    clean, resid = {}, {}
    for ti in range(len(rp.TEMPLATES)):
        group = [l for l in lines if l[0] == ti]
        for i in range(0, len(group), 24):
            chunk = group[i:i + 24]
            cache, logits, cap = rp.prefill(m, rp.encode(tok, [rp.prompt(*l[:2]) for l in chunk]), capture=True)
            outs = rp.greedy(m, tok, cache, logits)
            for j, (l, o) in enumerate(zip(chunk, outs)):
                clean[tuple(l)] = dict(line2=o, last=rp.last_word(o), ok=rp.rhymes(rp.last_word(o), l[1]))
                resid[tuple(l)] = torch.stack([c[j] for c in cap]).float().half().cpu().numpy()
    R = np.stack([resid[l] for l in clean])
    if not pilot:
        np.save(stem + "_resid.npy", R)
        json.dump({json.dumps(list(k)): v for k, v in clean.items()}, open(stem + "_clean.json", "w"))
L_ORDER = list(clean)  # row order of R
n_ok = sum(v["ok"] for v in clean.values())
n_copy = sum(v["ok"] and v["last"] == k[1] for k, v in clean.items())
print(f"clean rhymes that are copies of the line-1 word: {n_copy}/{n_ok}", flush=True)
print(f"{name}: clean rhyme rate {n_ok}/{len(clean)} = {n_ok/len(clean):.3f}  ({time.time()-t0:.0f}s)", flush=True)
if pilot:
    for l in list(clean)[:8]:
        print(l, repr(clean[l]["line2"]))
    sys.exit()

# ---- pairs among eligible lines (same template, different class); fixed batches so resume is bit-identical
elig = [l for l in LINES if l in clean and (clean[l]["ok"] or args.any_pairs)]  # LINES order, as before
pairs = [(a, b) for a in elig for b in elig if a[0] == b[0] and a[2] != b[2]]
pairs = rng.sample(pairs, min(NPAIRS, len(pairs)))
pairs.sort(key=lambda p: p[0][0])
batches = [g[i:i + BS] for ti in range(len(rp.TEMPLATES)) for g in [[p for p in pairs if p[0][0] == ti]] for i in range(0, len(g), BS)]
print(f"eligible lines {len(elig)}, pairs used {len(pairs)}, batches {len(batches)}", flush=True)

CONDS = ["NL", "NL_rand", "KV", "KV_rand"] + (["STATE", "STATE_rand", "STATE_rec", "STATE_rec_rand", "BOTH"] if hybrid else [])
jl = stem + ".pairs.jsonl"
recs = [json.loads(x) for x in open(jl)] if os.path.exists(jl) else []
done = {r["batch"] for r in recs}
assert all(len([r for r in recs if r["batch"] == b]) == len(batches[b]) for b in done), "corrupt pairs.jsonl; delete it"
for bi, sub in enumerate(batches):
    if bi in done:
        continue
    gen = torch.Generator().manual_seed(1000 + bi)  # per-batch seed: rand controls independent of resume point
    idT = rp.encode(tok, [rp.prompt(*t[:2]) for t, _ in sub])
    idS = rp.encode(tok, [rp.prompt(*s[:2]) for _, s in sub])
    _, _, capT = rp.prefill(m, idT, capture=True)
    _, _, capS = rp.prefill(m, idS, capture=True)
    cS, _, _ = rp.prefill(m, idS[:, :-1])  # cache patches: memory up to and incl. the line-1 word
    outs = {}
    for c in CONDS:
        if c.startswith("NL"):
            patch = capS if c == "NL" else [rp.rand_like_delta(t, s, gen) for t, s in zip(capT, capS)]
            cache, logits, _ = rp.prefill(m, idT, nl_patch=patch)
        else:
            cache, _, _ = rp.prefill(m, idT[:, :-1])
            kinds = ["state", "kv"] if c == "BOTH" else [c.split("_rand")[0].lower()]
            for k in kinds:
                rp.patch_cache(cache, cS, k, rand=c.endswith("_rand"), gen=gen)
            out = m(input_ids=idT[:, -1:], past_key_values=cache, use_cache=True)  # newline token on patched memory
            cache, logits = out.past_key_values, out.logits[:, -1]
        outs[c] = rp.greedy(m, tok, cache, logits)
    new = []
    for j, (t, s) in enumerate(sub):
        r = dict(batch=bi, T=t[1], S=s[1], tmpl=t[0])
        for c in CONDS:
            w = rp.last_word(outs[c][j])
            r[c] = dict(last=w, flip=rp.rhymes(w, s[1]) and not rp.rhymes(w, t[1]), keepT=rp.rhymes(w, t[1]), copyS=w == s[1])
        new.append(r)
    with open(jl, "a") as f:
        f.write("".join(json.dumps(r) + "\n" for r in new)); f.flush(); os.fsync(f.fileno())
    recs += new
    print(f"batch {bi + 1}/{len(batches)}  {len(recs)}/{len(pairs)} pairs  {time.time()-t0:.0f}s", flush=True)

summ = {c: dict(flip=wilson(sum(r[c]["flip"] for r in recs), len(recs)),
                keepT=wilson(sum(r[c]["keepT"] for r in recs), len(recs)),
                flip_nocopy=wilson(sum(r[c]["flip"] and not r[c]["copyS"] for r in recs), len(recs))) for c in CONDS}
for c, v in summ.items():
    print(f"{c:15s} flip {v['flip']}  flip_nocopy {v['flip_nocopy']}  keepT {v['keepT']}")

# ---- probe (secondary, weak): newline residual -> rhyme class of line-1 word; hold out words; shuffled-label control
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
X = R.astype(np.float32)
y = np.array([l[2] for l in L_ORDER]); g = np.array([l[1] for l in L_ORDER])
ysh = np.random.default_rng(0).permutation(y)
probe = {}
for L in range(X.shape[1]):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, C=0.1))
    a = cross_val_score(clf, X[:, L], y, groups=g, cv=GroupKFold(5)).mean()
    s = cross_val_score(clf, X[:, L], ysh, groups=g, cv=GroupKFold(5)).mean()
    probe[L] = (round(a, 3), round(s, 3))
print("probe (acc, shuffled) per layer:", probe, f"chance~{1/len(set(y)):.3f}", flush=True)

json.dump(dict(model=name, device=rp.DEV, dtype=str(rp.DTYPE), kernels=rp.kernel_info(), n_lines=len(clean),
               clean_rhyme=n_ok / len(clean), clean_copies=n_copy, n_elig=len(elig), n_pairs=len(recs),
               summary=summ, probe=probe, records=recs, clean={f"{k[0]}|{k[1]}": v for k, v in clean.items()},
               seconds=time.time() - t0), open(stem + ".json", "w"), indent=1)
