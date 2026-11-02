"""Classical arm: SLSQP (analytic Jacobians) multistart + basin hopping, anytime.

usage: python baseline.py PROBLEM SEED SECONDS OUT.json [p_fresh sigma]
"""
import json, sys, time
import numpy as np
from scipy.optimize import minimize
from problems import PROBLEMS, score, better


def circ_solve(z0, n):
    i, j = np.triu_indices(n, 1)
    m = len(i)

    def f(z): return -z[2 * n:].sum()
    def fg(z):
        g = np.zeros(3 * n); g[2 * n:] = -1; return g

    def cons(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        pair = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 - (r[i] + r[j]) ** 2
        return np.concatenate([pair, x - r, 1 - x - r, y - r, 1 - y - r])

    def jac(z):
        x, y, r = z[:n], z[n:2 * n], z[2 * n:]
        J = np.zeros((m + 4 * n, 3 * n)); k = np.arange(m)
        dx, dy, sr = x[i] - x[j], y[i] - y[j], r[i] + r[j]
        J[k, i] = 2 * dx; J[k, j] = -2 * dx
        J[k, n + i] = 2 * dy; J[k, n + j] = -2 * dy
        J[k, 2 * n + i] = -2 * sr; J[k, 2 * n + j] = -2 * sr
        a = np.arange(n)
        for b, (cx, cy, cr) in enumerate([(1, 0, -1), (-1, 0, -1), (0, 1, -1), (0, -1, -1)]):
            row = m + b * n + a
            J[row, a] = cx; J[row, n + a] = cy; J[row, 2 * n + a] = cr
        return J

    bnds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
    res = minimize(f, z0, jac=fg, method="SLSQP", bounds=bnds,
                   constraints=[dict(type="ineq", fun=cons, jac=jac)],
                   options=dict(maxiter=600, ftol=1e-13))
    z = res.x
    return repair_circles(np.stack([z[:n], z[n:2 * n], z[2 * n:]], 1))


def repair_circles(s):
    """Shrink radii until exactly feasible (no tolerance used)."""
    s = s.copy(); n = len(s)
    s[:, :2] = np.clip(s[:, :2], 0, 1)
    i, j = np.triu_indices(n, 1)
    for _ in range(50):
        x, y = s[:, 0], s[:, 1]
        s[:, 2] = np.clip(np.minimum.reduce([s[:, 2], x, 1 - x, y, 1 - y]), 0, None)
        d = np.hypot(x[i] - x[j], y[i] - y[j])
        ov = s[i, 2] + s[j, 2] - d
        bad = ov > 0
        if not bad.any():
            break
        np.subtract.at(s[:, 2], i[bad], ov[bad] / 2 + 1e-13)
        np.subtract.at(s[:, 2], j[bad], ov[bad] / 2 + 1e-13)
    s[:, 2] = np.clip(s[:, 2], 0, None)
    return s


def ratio_solve(p0, n):
    i, j = np.triu_indices(n, 1)
    m = len(i)
    def f(z): return z[-1]
    def fg(z):
        g = np.zeros(2 * n + 1); g[-1] = 1; return g
    def d2(z):
        x, y = z[:n], z[n:2 * n]; return (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2
    def cons(z): q = d2(z); return np.concatenate([q - 1, z[-1] - q])
    def jac(z):
        x, y = z[:n], z[n:2 * n]; k = np.arange(m)
        J = np.zeros((m, 2 * n + 1))
        J[k, i] = 2 * (x[i] - x[j]); J[k, j] = -J[k, i]
        J[k, n + i] = 2 * (y[i] - y[j]); J[k, n + j] = -J[k, n + i]
        J2 = -J.copy(); J2[:, -1] = 1
        return np.vstack([J, J2])
    p = p0 - p0.mean(0)
    dd = np.hypot(*(p[i] - p[j]).T)
    p = p / dd.min()
    z0 = np.concatenate([p[:, 0], p[:, 1], [(dd.max() / dd.min()) ** 2]])
    res = minimize(f, z0, jac=fg, method="SLSQP",
                   constraints=[dict(type="ineq", fun=cons, jac=jac)],
                   options=dict(maxiter=600, ftol=1e-14))
    return np.stack([res.x[:n], res.x[n:2 * n]], 1)


def run(name, seed, seconds, p_fresh=0.2, sigma=0.1):
    P = PROBLEMS[name]; n = P["n"]
    rng = np.random.default_rng(seed)
    best, best_sol, curve, t0, it = None, None, [], time.time(), 0
    while time.time() - t0 < seconds:
        it += 1
        if best_sol is None or rng.random() < p_fresh:
            if P["kind"] == "circles":
                start = np.column_stack([rng.random((n, 2)), np.full(n, 0.01)])
            else:
                start = rng.random((n, 2))
        else:  # basin hop: jitter all + relocate a few
            start = best_sol.copy()
            scale = 1.0 if P["kind"] == "circles" else np.ptp(start)
            start[:, :2] += rng.normal(0, scale * 10 ** rng.uniform(-2.5, np.log10(sigma)), (n, 2))
            k = rng.integers(0, 3)
            idx = rng.choice(n, k, replace=False)
            if P["kind"] == "circles":
                start[idx, :2] = rng.random((k, 2)); start[:, 2] *= 0.9
                start[:, :2] = np.clip(start[:, :2], 0, 1)
        try:
            if P["kind"] == "circles":
                z0 = np.concatenate([start[:, 0], start[:, 1], start[:, 2]])
                sol = circ_solve(z0, n)
            else:
                sol = ratio_solve(start, n)
        except Exception:
            continue
        ok, s, _ = score(name, sol)
        if ok and better(name, s, best):
            best, best_sol = s, sol
            curve.append((time.time() - t0, s))
    return dict(problem=name, seed=seed, seconds=seconds, iters=it, best=best,
                curve=curve, solution=best_sol.tolist(), p_fresh=p_fresh, sigma=sigma)


if __name__ == "__main__":
    name, seed, secs, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    extra = [float(v) for v in sys.argv[5:7]]
    r = run(name, seed, secs, *extra)
    json.dump(r, open(out, "w"))
    print(name, seed, r["iters"], r["best"])
