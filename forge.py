"""kernel-forge harness: targets, baselines, kernel building, correctness, paired timing.

CLI (used by evolve.py so a bad kernel can only crash / hang a subprocess):
  python forge.py eval  <target> <spec.json>          -> JSON fitness on stdout (search mode)
  python forge.py final <target> <spec.json> [hand]   -> JSON final report (1000 checks, 60 reps)
"""
import json, math, sys, time, fcntl, os
import numpy as np
import mlx.core as mx
import mlx.nn as nn

F16 = mx.float16
EPS = 1e-5
GS = 64
LOCK = "/tmp/kernel_forge_gpu.lock"  # serialize GPU work across parallel target processes

# ---------------------------------------------------------------- targets
TIMING = {
    "A": [(1, 2048), (1, 4096), (32, 4096), (128, 4096)],          # (rows, H)
    "B": [(1, 8192), (1, 14336), (32, 14336), (128, 11008)],        # (rows, I)
    "C": [(4096, 4096), (4096, 14336), (14336, 4096), (2048, 151936)],  # (K, N)
}
ODD = {
    "A": [(1, 1), (3, 1000), (7, 4097), (5, 333)],
    "B": [(1, 1), (3, 999), (2, 14335), (5, 7)],
    "C": [(64, 1), (128, 7), (4096, 4097), (192, 333)],
}
DIMS = {"A": ("rows", "H"), "B": ("rows", "I"), "C": ("K", "N")}
IN_NAMES = {"A": ["x", "r", "w"], "B": ["g", "u"], "C": ["x", "wq", "scales", "biases"]}
OUT_NAMES = {"A": ["h", "y"], "B": ["out"], "C": ["out"]}


def make_inputs(t, shape, seed):
    k = mx.random.key(seed)
    k1, k2, k3 = mx.random.split(k, 3)
    if t == "A":
        rows, H = shape
        return [mx.random.normal((rows, H), key=k1).astype(F16),
                mx.random.normal((rows, H), key=k2).astype(F16),
                (1 + 0.1 * mx.random.normal((H,), key=k3)).astype(F16)]
    if t == "B":
        rows, I = shape
        return [(3 * mx.random.normal((rows, I), key=k1)).astype(F16),
                mx.random.uniform(-1, 1, (rows, I), key=k2).astype(F16)]
    K, N = shape
    w = (0.02 * mx.random.normal((N, K), key=k1)).astype(F16)
    wq, s, b = mx.quantize(w, group_size=GS, bits=4)
    return [mx.random.normal((1, K), key=k2).astype(F16), wq, s, b]


def reference(t, inp):
    f = lambda a: np.array(a.astype(mx.float32)) if a.dtype != mx.uint32 else np.array(a)
    if t == "A":
        x, r, w = map(f, inp)
        h = (x + r).astype(np.float16).astype(np.float32)  # h is stored fp16
        y = h / np.sqrt((h * h).mean(-1, keepdims=True) + EPS) * w
        return [h, y]
    if t == "B":
        g, u = map(f, inp)
        return [g / (1 + np.exp(-g)) * u]
    x, wq, s, b = inp
    W = np.array(mx.dequantize(wq, s, b, group_size=GS, bits=4).astype(mx.float32))
    return [f(x) @ W.T]


def check(t, outs, refs):
    for o, ref in zip(outs, refs):
        o = np.array(o.astype(mx.float32))
        if o.shape != ref.shape or not np.all(np.isfinite(o)):
            return False
        if t == "C":
            tol = 0.02 * np.abs(ref) + 0.02 * ref.std() + 1e-3
        else:
            tol = 1e-2 + 1e-2 * np.abs(ref)
        if not np.all(np.abs(o - ref) <= tol):
            return False
    return True


def step(t, inp, outs):
    """Next inputs in the timing chain (dependency like in a real model)."""
    if t == "A":
        return [outs[1], outs[0], inp[2]]
    if t == "B":
        return [outs[0], inp[1]]
    if outs[0].shape == inp[0].shape:
        return [outs[0]] + inp[1:]
    return inp


# ---------------------------------------------------------------- baselines
def builtin(t):
    if t == "A":
        def f(x, r, w):
            h = x + r
            return [h, mx.fast.rms_norm(h, w, EPS)]
    elif t == "B":
        def f(g, u):
            return [nn.silu(g) * u]
    else:
        def f(x, wq, s, b):
            return [mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=GS, bits=4)]
    return f


def compiled(t):
    if t == "C":
        def f(x, wq, s, b):
            return [x @ mx.dequantize(wq, s, b, group_size=GS, bits=4).T]
        return mx.compile(f)
    return mx.compile(builtin(t))


# ---------------------------------------------------------------- LLM/hand kernels
def build(t, spec):
    """spec = {source, header?, threadgroup:[x,y,z], grid:[expr,expr,expr]}; grid exprs use DIMS names + cdiv."""
    kern = mx.fast.metal_kernel(name=f"cand_{t}", input_names=IN_NAMES[t], output_names=OUT_NAMES[t],
                                source=spec["source"], header=spec.get("header", ""))
    tg = tuple(int(v) for v in spec["threadgroup"])
    assert len(tg) == 3 and math.prod(tg) <= 1024

    cache = {}  # per-shape launch config, so the timed path costs the same as a hand-written wrapper

    def launch_cfg(key):
        if key not in cache:
            d = {"K": key[0], "N": key[1]} if t == "C" else dict(zip(DIMS[t], key))
            env = dict(d, cdiv=lambda a, b: -(-a // b), min=min, max=max)
            grid = tuple(int(eval(str(e), {"__builtins__": {}}, env)) for e in spec["grid"])
            oshape = [(1, d["N"])] if t == "C" else [key] * len(OUT_NAMES[t])
            cache[key] = (list(d.items()), grid, oshape, [F16] * len(oshape))
        return cache[key]

    def f(*inp):
        key = (inp[0].shape[1], inp[1].shape[0]) if t == "C" else inp[0].shape
        tmpl, grid, oshape, odt = launch_cfg(key)
        return kern(inputs=list(inp), template=tmpl, grid=grid, threadgroup=tg,
                    output_shapes=oshape, output_dtypes=odt)
    return f


# ---------------------------------------------------------------- timing
def sample(t, f, inp, K):
    s, allo = inp, []
    t0 = time.perf_counter()
    for _ in range(K):
        o = f(*s)
        s = step(t, s, o)
        allo += o
    mx.eval(allo)  # evaluate every output: unevaluated lazy outputs would be skipped by MLX
    return (time.perf_counter() - t0) / K, s, o


def time_shape(t, fns, shape, reps, K=32, seed=0):
    """Interleaved timing of several fns (rotating order). Returns {name: list of per-call seconds}."""
    inp = make_inputs(t, shape, seed)
    mx.eval(inp)
    names = list(fns)
    for n in names:  # warmup
        for _ in range(3):
            sample(t, fns[n], inp, K)
    res = {n: [] for n in names}
    for i in range(reps):
        order = names[i % len(names):] + names[:i % len(names)]
        if (i // len(names)) % 2:
            order = order[::-1]
        for n in order:
            res[n].append(sample(t, fns[n], inp, K)[0])
    return res


def speedup(base, cand, nboot=2000, seed=0):
    r = np.array(base) / np.array(cand)
    rng = np.random.default_rng(seed)
    boots = np.median(r[rng.integers(0, len(r), (nboot, len(r)))], axis=1)
    return float(np.median(r)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def correctness(t, f, n_checks, seed0=1000):
    shapes = TIMING[t] + ODD[t]
    for i in range(n_checks):
        shape = shapes[i % len(shapes)]
        inp = make_inputs(t, shape, seed0 + i)
        outs = f(*inp)
        mx.eval(outs)
        if not check(t, outs, reference(t, inp)):
            return False, f"fail check {i} shape {shape}"
    return True, f"{n_checks}/{n_checks} ok"


def evaluate(t, spec, n_checks, reps, K, final=False):
    f = build(t, spec)
    with open(LOCK, "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        ok, msg = correctness(t, f, n_checks)
        if not ok:
            return {"correct": False, "msg": msg}
        out = {"correct": True, "msg": msg, "shapes": []}
        for shape in TIMING[t]:
            fns = {"builtin": builtin(t), "compiled": compiled(t), "cand": f}
            res = time_shape(t, fns, shape, reps, K)
            med = {n: float(np.median(v)) for n, v in res.items()}
            best = min(("builtin", "compiled"), key=lambda n: med[n])
            s, lo, hi = speedup(res[best], res["cand"])
            # timed-run output check: last chain output must still match reference
            inp = make_inputs(t, shape, 777)
            _, s_in, o = sample(t, f, inp, 1)
            row = {"shape": shape, "us": {n: 1e6 * v for n, v in med.items()}, "best_base": best,
                   "speedup": s, "ci": [lo, hi], "timed_out_ok": check(t, o, reference(t, inp))}
            if final:
                row["vs_builtin"] = speedup(res["builtin"], res["cand"])[0]
                row["vs_compiled"] = speedup(res["compiled"], res["cand"])[0]
            out["shapes"].append(row)
        out["fitness"] = float(np.exp(np.mean([np.log(r["speedup"]) for r in out["shapes"]])))
        return out


if __name__ == "__main__":
    mode, t, path = sys.argv[1:4]
    spec = json.load(open(path))
    if mode == "eval":
        r = evaluate(t, spec, n_checks=16, reps=15, K=16)
    else:
        r = evaluate(t, spec, n_checks=1000, reps=60, K=32, final=True)
    print("RESULT " + json.dumps(r))
