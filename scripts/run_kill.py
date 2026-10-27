"""Kill experiment: do linear layers in released hybrids carry long-range information?

Run:    python scripts/run_kill.py run --model Qwen/Qwen3.5-0.8B
Report: python scripts/run_kill.py report

Arms per window W: full | lin@W (linear mixers windowed) | attn@W (attention windowed) | trunc@W (whole
model sees only W/2..W tokens: the dumbest strong baseline and the validity control).
Pre-registered gate (written before any result exists) is in `verdict`.
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hybrid_locality.evals import FAR, exact_match, haystack, longppl, make_task, target_logprobs  # noqa: E402
from hybrid_locality.window import intervene  # noqa: E402

MODELS = ["Qwen/Qwen3.5-0.8B", "tiiuae/Falcon-H1-0.5B-Base", "tiiuae/Falcon-H1-1.5B-Base",
          "Zyphra/Zamba2-1.2B", "ibm-granite/granite-4.0-h-micro"]
TASKS = ["single", "multikey", "vt"]
OUT = Path(__file__).resolve().parents[1] / "results" / "kill.jsonl"


def arms(windows):
    yield "full", None
    for W in windows:
        for a in ("lin", "attn", "trunc"):
            yield a, W


def score(model, arm, W, fn):
    if arm in ("full", "trunc"):
        return fn(trunc=W)
    with intervene(model, **{arm: W}):
        return fn(trunc=None)


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
    cfg = getattr(model.config, "get_text_config", lambda: model.config)()
    max_len = getattr(cfg, "max_position_embeddings", None) or max(args.lengths)
    lengths = [L for L in args.lengths if L <= max_len]
    print(f"{args.model}: lengths {lengths} (max_position_embeddings={max_len}, device={device}, dtype={dtype})", flush=True)
    books = haystack(tok)
    hay = [t for b in books for t in b + _sep(tok)]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("a") as f:
        put = lambda **r: (f.write(json.dumps({"model": args.model, **r}) + "\n"), f.flush())
        for L in lengths:
            for task in TASKS:
                rng = random.Random(f"{task}-{L}")  # identical items across models and arms
                items = [make_task(task, tok, hay, L, rng) for _ in range(args.n)]
                for arm, W in arms(args.windows):
                    for i, (ids, na) in enumerate(items):
                        ok = score(model, arm, W, lambda trunc: exact_match(model, ids.to(device), na, trunc))
                        put(kind="ruler", task=task, L=L, arm=arm, W=W, i=i, ok=ok)
                    print(f"  L={L} {task} {arm}@{W} done", flush=True)
            docs = [b[:L] for b in books if len(b) >= L][: args.docs]
            for i, d in enumerate(docs):
                ids = torch.tensor([d]).to(device)
                full, short = target_logprobs(model, ids), target_logprobs(model, ids, trunc=FAR)
                for arm, W in arms(args.windows):
                    lp = score(model, arm, W, lambda trunc: target_logprobs(model, ids, trunc))
                    put(kind="ppl", L=L, arm=arm, W=W, i=i, **longppl(lp, full, short))
            print(f"  L={L} longppl done ({len(docs)} docs)", flush=True)


def _sep(tok):
    return tok("\n\n", add_special_tokens=False).input_ids


def boot_diff(a, b, n=2000, seed=0):
    """Paired bootstrap 95% CI of mean(a) - mean(b)."""
    g, d = torch.Generator().manual_seed(seed), torch.tensor(a, dtype=torch.float) - torch.tensor(b, dtype=torch.float)
    m = d[torch.randint(len(d), (n, len(d)), generator=g)].mean(1)
    return float(d.mean()), float(m.quantile(0.025)), float(m.quantile(0.975))


def summarize(rows, W):
    """Per model: RULER accuracy drop (points) for each arm at W, and relative LongPPL key-NLL rise."""
    ruler, ppl = defaultdict(dict), defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] == "ruler" and r["W"] in (None, W):
            ruler[r["model"]][(r["arm"], r["task"], r["L"], r["i"])] = r["ok"]
        if r["kind"] == "ppl" and r["W"] in (None, W) and r["n_key"] > 0:
            ppl[r["model"]][r["arm"]].append(r["key_nll"])
    out = {}
    for m, d in ruler.items():
        base = {k[1:]: v for k, v in d.items() if k[0] == "full"}
        s = {"full_acc": 100 * sum(base.values()) / max(1, len(base))}
        for arm in ("lin", "attn", "trunc"):
            keys = [k for k in base if (arm, *k) in d]
            if keys:
                s[f"{arm}_drop"] = tuple(100 * x for x in boot_diff([base[k] for k in keys], [d[(arm, *k)] for k in keys]))
            if ppl[m]["full"] and ppl[m][arm]:
                f, a = sum(ppl[m]["full"]) / len(ppl[m]["full"]), sum(ppl[m][arm]) / len(ppl[m][arm])
                s[f"{arm}_keynll_rise_pct"] = 100 * (a - f) / f
        out[m] = s
    return out


def verdict(summary, min_models=3):
    """Pre-registered rule at W=1024. A model counts only if Full accuracy >= 50% and Trunc drops >= 20 points
    (otherwise the tasks do not test long context for that model). Then:
      H1 (linear layers are local mixers): lin drop <= 2 pts AND lin key-NLL rise <= 3% AND attn drop >= 20 pts
      H0 (linear layers carry long-range info): lin drop >= 10 pts
    H1/H0 if >= min_models valid models agree; INVALID if < min_models valid; else MIXED (per-layer map is the paper).
    Note: LongPPL rise is measured on key-token NLL; 3% of NLL, not of perplexity."""
    valid = {m: s for m, s in summary.items() if s["full_acc"] >= 50 and s.get("trunc_drop", (0,))[0] >= 20}
    if len(valid) < min_models:
        return "INVALID", valid
    h1 = sum(s["lin_drop"][0] <= 2 and s.get("lin_keynll_rise_pct", 1e9) <= 3 and s["attn_drop"][0] >= 20 for s in valid.values())
    h0 = sum(s["lin_drop"][0] >= 10 for s in valid.values())
    return ("H1" if h1 >= min_models else "H0" if h0 >= min_models else "MIXED"), valid


def report(args):
    rows = [json.loads(l) for l in OUT.open()]
    for W in sorted({r["W"] for r in rows if r["W"]}):
        print(f"\n=== W={W}")
        summ = summarize(rows, W)
        for m, s in summ.items():
            print(f"{m:40s} " + "  ".join(f"{k}={v[0]:.1f}[{v[1]:.1f},{v[2]:.1f}]" if isinstance(v, tuple) else f"{k}={v:.1f}" for k, v in s.items()))
        if W == 1024:
            v, valid = verdict(summ)
            print(f"\nVERDICT (W=1024, pre-registered): {v}   valid models: {list(valid)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    r.add_argument("--windows", type=int, nargs="+", default=[64, 256, 1024, 4096])
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--docs", type=int, default=20)
    sub.add_parser("report")
    a = p.parse_args()
    run(a) if a.cmd == "run" else report(a)
