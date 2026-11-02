"""Problem definitions + exact verifiers. Same verifier for both arms.

Circle packing: solution = array (n,3) of (x, y, r). Feasible iff all r>=0,
r_i <= x_i <= 1-r_i (same for y) and ||c_i-c_j|| >= r_i+r_j, each up to TOL.
Score = sum(r) (maximize).
Min/max distance: solution = array (16,2). Score = (dmax/dmin)^2 (minimize).
"""
import numpy as np

TOL = 1e-9

# Best-known values (see README prior art).
PROBLEMS = {
    "circ26": dict(kind="circles", n=26, best=2.635983, maximize=True),   # ShinkaEvolve / Gideoni SCS 2.63598-2.63590
    "circ32": dict(kind="circles", n=32, best=2.93957, maximize=True),    # Xpress/SCIP, arXiv 2601.05943
    "ratio16": dict(kind="ratio", n=16, best=12.88923, maximize=False),   # FM Agent / 2601.05943 (12.88924)
}


def score(name, sol):
    """Return (feasible: bool, score: float or None, msg)."""
    p = PROBLEMS[name]
    try:
        a = np.asarray(sol, dtype=np.float64)
    except Exception as e:
        return False, None, f"not numeric: {e}"
    if not np.all(np.isfinite(a)):
        return False, None, "non-finite"
    if p["kind"] == "circles":
        if a.shape != (p["n"], 3):
            return False, None, f"shape {a.shape}"
        x, y, r = a.T
        if r.min() < 0:
            return False, None, "negative radius"
        cont = max((r - x).max(), (x + r - 1).max(), (r - y).max(), (y + r - 1).max())
        if cont > TOL:
            return False, None, f"containment violated by {cont:.3g}"
        i, j = np.triu_indices(p["n"], 1)
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        ov = (r[i] + r[j] - d).max()
        if ov > TOL:
            return False, None, f"overlap {ov:.3g}"
        return True, float(r.sum()), "ok"
    if a.shape != (p["n"], 2):
        return False, None, f"shape {a.shape}"
    i, j = np.triu_indices(p["n"], 1)
    d = np.hypot(a[i, 0] - a[j, 0], a[i, 1] - a[j, 1])
    if d.min() <= 1e-6 * max(d.max(), 1e-300):
        return False, None, "coincident points"
    return True, float((d.max() / d.min()) ** 2), "ok"


def frac_of_best(name, s):
    p = PROBLEMS[name]
    return s / p["best"] if p["maximize"] else p["best"] / s


def better(name, a, b):
    """True if score a strictly better than b (None = worst)."""
    if a is None:
        return False
    if b is None:
        return True
    return a > b if PROBLEMS[name]["maximize"] else a < b
