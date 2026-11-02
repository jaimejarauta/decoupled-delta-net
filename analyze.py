"""H1 (early-warning) and H2 (controller vs baselines) analysis, exactly as pre-registered in README.md."""
import json, sys
import numpy as np
from grok import steps_to

SIGNALS = ["wnorm", "erank", "train_loss", "gnorm"]
KAPPAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
WD_GRID = [0.1, 0.3, 1.0, 3.0]
T = 10000  # README amendment A9 (A8's 6000 withdrawn); campaign.py may override
RUNS = "results"


def load(name):
    return json.load(open(f"{RUNS}/{name}.json"))


def arm_key(r, i):
    c = r["ctrl"][i]
    if c is not None:
        return "CTRL:" + json.dumps(c, sort_keys=True)
    if r["gf"][i] > 0:
        return "B2 grokfast wd=0.005" if r["wds"][i] == 0.005 else f"B3 grokfast+wd={r['wds'][i]}"
    return f"B1 wd={r['wds'][i]}"


def arms(r):
    """{arm: array of steps-to-95% over seeds} (inf = censored)."""
    s = steps_to(r["log"])
    s[s > T] = np.inf  # amendment A8: budget T
    out = {}
    for i in range(len(s)):
        out.setdefault(arm_key(r, i), []).append(s[i])
    return {k: np.array(v) for k, v in out.items()}


# ---------------- H1 ----------------
def h1_events(r, idx):
    """Per run: (t_mem, t_grok, {signal: (steps, causal smoothed log-signal)}) for model columns idx."""
    L = r["log"]
    st = np.array(L["step"])
    tr, te = np.array(L["train_acc"]), np.array(L["test_acc"])
    out = []
    for i in idx:
        if not (tr[:, i] >= 0.99).any() or not (te[:, i] >= 0.95).any():
            continue
        t_mem, t_grok = st[(tr[:, i] >= 0.99).argmax()], st[(te[:, i] >= 0.95).argmax()]
        if t_grok - t_mem < 500:
            continue
        sig = {}
        for s in SIGNALS:
            x = np.log(np.maximum(np.array(L[s])[:, i], 1e-12))
            c = np.cumsum(np.insert(x, 0, 0))
            k = np.arange(1, len(x) + 1)
            lo = np.maximum(k - 5, 0)
            sig[s] = (c[k] - c[lo]) / (k - lo)  # trailing mean over last <=5 logs
        out.append((t_mem, t_grok, st, sig))
    return out


def fire(t_mem, st, x, kappa):
    j0 = np.searchsorted(st, t_mem)
    hit = np.nonzero((st > t_mem) & (np.abs(x - x[j0]) >= kappa))[0]
    return st[hit[0]] if len(hit) else np.inf


def in_window(ev, s, kappa):
    t_mem, t_grok, st, sig = ev
    D = t_grok - t_mem
    tf = fire(t_mem, st, sig[s], kappa)
    return t_mem + 0.2 * D <= tf <= t_grok - 0.2 * D, (t_grok - tf) / D


def h1_score(evs, s, kappa):
    return np.mean([in_window(e, s, kappa)[0] for e in evs]) if evs else np.nan


def b1_idx(r):
    return [i for i in range(len(r["seeds"])) if r["ctrl"][i] is None and r["gf"][i] == 0]


# ---------------- H2 ----------------
def boot_ratio(a, b, T, n=10000, seed=0):
    g = np.random.default_rng(seed)
    a, b = np.minimum(a, 2 * T), np.minimum(b, 2 * T)
    ra = np.median(g.choice(a, (n, len(a))), 1)
    rb = np.median(g.choice(b, (n, len(b))), 1)
    return np.median(a) / np.median(b), np.percentile(ra / rb, [2.5, 97.5])


def h2_task(r, T):
    A = arms(r)
    base = {k: v for k, v in A.items() if not k.startswith("CTRL")}
    med = {k: np.median(np.minimum(v, 2 * T)) for k, v in A.items()}
    best = min(base, key=lambda k: med[k])
    ctrl = [k for k in A if k.startswith("CTRL")]
    assert len(ctrl) == 1
    R, ci = boot_ratio(A[ctrl[0]], A[best], T)
    return dict(arms={k: (med[k], A[k].tolist()) for k in A}, best=best, R=R, ci=ci.tolist(),
                ctrl_med=med[ctrl[0]], best_med=med[best])


# ---------------- pipeline helpers ----------------
def pick_min(A):
    """Arm with lowest median steps (censored = 2T); ties -> first listed."""
    meds = {k: np.median(np.minimum(v, 2 * T)) for k, v in A.items()}
    return min(meds, key=lambda k: (meds[k], list(meds).index(k))), meds


def h1_select(dev):
    """Signal and kappa chosen on dev constant-WD runs; ties -> smaller kappa, then signal order S1..S4."""
    evs = h1_events(dev, b1_idx(dev))
    table = {(s, k): h1_score(evs, s, k) for s in SIGNALS for k in KAPPAS}
    best = max(table, key=lambda sk: (np.nan_to_num(table[sk], nan=-1), -KAPPAS.index(sk[1])))
    return best, table, len(evs)


def h1_heldout(held, s, kappa):
    """held: list of run dicts; pooled over their constant-WD (B1) columns. PASS iff >=1 eligible and >=80% in window."""
    evs = [e for r in held for e in h1_events(r, b1_idx(r))]
    score = h1_score(evs, s, kappa)
    leads = [in_window(e, s, kappa)[1] for e in evs]
    return dict(n=len(evs), score=score, leads=leads, PASS=bool(evs and score >= 0.8))


def h2_verdict(rows, need=3):
    """rows: [(R, ci_upper)] per task. PASS / MIXED / FAIL as pre-registered (>= need tasks)."""
    if sum(R <= 0.5 and hi < 1 for R, hi in rows) >= need:
        return "PASS"
    return "MIXED" if sum(R < 1 and hi < 1 for R, hi in rows) >= need else "FAIL"


if __name__ == "__main__":
    exec(sys.argv[1])
