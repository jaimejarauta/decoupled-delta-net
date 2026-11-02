"""LLM evolution loop: python evolve.py <target> <n_calls>. Archive + logs in runs/."""
import json, os, re, subprocess, sys, time, fcntl

ROOT = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(ROOT, "runs")
PY = sys.executable
CAP = 200  # hard global LLM-call budget
MODEL = "sonnet"

COMMON = """You write ONE Apple-Silicon Metal compute kernel body for MLX's mx.fast.metal_kernel (Apple M4 GPU).
Rules of mx.fast.metal_kernel:
- You write only the BODY of the kernel function. MLX generates the signature from the input/output names below.
- Buffers: inputs are `const device float16_t*`, BUT MLX passes very small inputs (a few elements) as `const constant float16_t*`, so a C-style cast like `(const device half4*)(g + i)` fails to compile for tiny shapes. Use `auto` for derived pointers, and for vector loads use a templated helper in the header (template<typename P> ... ) or the address space of the input (e.g. `decltype(...)`/ scalar loads), so both address spaces compile, outputs `device float16_t*`. float16_t == half. Row-contiguous.
- Shape constants are provided as compile-time template ints (listed below); use them directly.
- You may use these attribute names in the body and MLX adds them to the signature: thread_position_in_grid, threadgroup_position_in_grid, thread_position_in_threadgroup, threads_per_threadgroup, thread_index_in_simdgroup, simdgroup_index_in_threadgroup (uint3 for the first four, uint for the last two).
- `threadgroup` arrays may be declared inside the body. simd_sum/simd_max etc. are available. Helper functions go in an optional header.
- grid = TOTAL number of threads per dimension (dispatchThreads semantics, NOT number of threadgroups). threadgroup size product <= 1024.
- grid is given as 3 Python integer expressions using the shape names below and cdiv(a,b) (ceil div), min, max.
- Must be correct for ALL shapes including tiny/odd ones (e.g. H=1, 333, 4097). Correctness is checked on fresh random inputs vs an fp32 reference.
- Goal: minimum wall time per call at the timing shapes, beating MLX's built-in and mx.compile baselines.

Answer format, exactly:
```metal
<kernel body>
```
optional:
```header
<helper functions>
```
```json
{"threadgroup": [tx, ty, tz], "grid": ["expr_x", "expr_y", "expr_z"], "idea": "<one sentence>"}
```
"""

TASK = {
    "A": """TARGET A: fused residual add + RMSNorm (decode / small batch).
Inputs: x (rows,H) half, r (rows,H) half, w (H,) half. Outputs: h (rows,H) half, y (rows,H) half.
h = x + r (rounded to half);  y = h / sqrt(mean(h^2 over the row) + 1e-5) * w   (compute in float).
Template ints: rows, H. Timing shapes (rows,H): (1,2048), (1,4096), (32,4096), (128,4096).
Baselines: MLX `h = x + r; mx.fast.rms_norm(h, w, 1e-5)` and the same under mx.compile.""",
    "B": """TARGET B: SwiGLU activation.
Inputs: g (rows,I) half, u (rows,I) half. Output: out (rows,I) half. out = silu(g) * u = g / (1 + exp(-g)) * u (compute in float).
Template ints: rows, I. Timing shapes (rows,I): (1,8192), (1,14336), (32,14336), (128,11008).
Baselines: MLX `nn.silu(g) * u` and the same under mx.compile (which fuses it into one kernel).""",
    "C": """TARGET C: 4-bit affine dequantize + matrix-vector product for batch-1 LLM decode (out = x @ W^T).
Inputs: x (1,K) half; wq (N, K/8) uint32 (`const device uint32_t*`); scales (N, K/64) half; biases (N, K/64) half. Output: out (1,N) half.
Quantization (MLX affine, group_size 64, bits 4): element W[n,k] = scales[n,k/64] * q + biases[n,k/64], where
q = (wq[n*(K/8) + k/8] >> (4*(k%8))) & 0xF   (low nibble first).  out[n] = sum_k x[k]*W[n,k] (accumulate in float).
K is always a multiple of 64; N can be any value >= 1 (e.g. 7, 333, 4097).
Template ints: K, N. Timing shapes (K,N): (4096,4096), (4096,14336), (14336,4096), (2048,151936).
Baselines: MLX `mx.quantized_matmul(x, wq, scales, biases, transpose=True, group_size=64, bits=4)` (highly tuned) and mx.compile(x @ dequantize(W).T).""",
}


def count_and_reserve():
    path = os.path.join(RUNS, "llm_calls.count")
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        n = int(open(path).read()) if os.path.exists(path) else 0
        if n >= CAP:
            return None
        open(path, "w").write(str(n + 1))
        return n + 1


def call_llm(prompt, target):
    n = count_and_reserve()
    if n is None:
        return None
    t0 = time.time()
    p = subprocess.run(["claude", "-p", "--model", MODEL], input=prompt, capture_output=True,
                       text=True, cwd="/tmp", timeout=600)
    rec = {"call": n, "target": target, "model": MODEL, "t": t0, "secs": time.time() - t0,
           "prompt": prompt, "response": p.stdout, "stderr": p.stderr[-2000:]}
    with open(os.path.join(RUNS, "llm_log.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")
    return p.stdout


def parse(resp):
    src = re.search(r"```metal\s*\n(.*?)```", resp, re.S)
    hdr = re.search(r"```header\s*\n(.*?)```", resp, re.S)
    js = re.search(r"```json\s*\n(.*?)```", resp, re.S)
    if not (src and js):
        raise ValueError("missing ```metal or ```json block")
    cfg = json.loads(js.group(1))
    return {"source": src.group(1), "header": hdr.group(1) if hdr else "",
            "threadgroup": cfg["threadgroup"], "grid": cfg["grid"], "idea": cfg.get("idea", "")}


def evaluate(target, spec, tag):
    path = os.path.join(RUNS, target, f"{tag}.json")
    json.dump(spec, open(path, "w"), indent=1)
    try:
        p = subprocess.run([PY, os.path.join(ROOT, "forge.py"), "eval", target, path],
                           capture_output=True, text=True, timeout=1200, cwd=ROOT)
    except subprocess.TimeoutExpired:
        return {"correct": False, "msg": "timeout"}
    m = re.search(r"^RESULT (.*)$", p.stdout, re.M)
    if not m:
        err = p.stderr or p.stdout
        err = err[err.find("Error"):] if "Error" in err else err[-1500:]  # skip the python traceback
        err = err[:1500]
        return {"correct": False, "msg": "compile/runtime error: " + err}
    return json.loads(m.group(1))


def fmt(entry):
    r = entry["result"]
    rows = "; ".join(f"{tuple(s['shape'])}: cand {s['us']['cand']:.1f}us vs builtin {s['us']['builtin']:.1f}us, "
                     f"compiled {s['us']['compiled']:.1f}us (speedup vs best {s['speedup']:.2f}x)" for s in r["shapes"])
    sp = entry["spec"]
    return (f"--- kernel {entry['tag']} | fitness (geomean speedup vs best baseline) {r['fitness']:.3f} | idea: {sp.get('idea','')}\n"
            f"timings: {rows}\nthreadgroup={sp['threadgroup']} grid={sp['grid']}\n"
            f"```metal\n{sp['source']}```\n" + (f"```header\n{sp['header']}```\n" if sp.get("header") else ""))


def main(target, n_calls):
    os.makedirs(os.path.join(RUNS, target), exist_ok=True)
    arch_path = os.path.join(RUNS, target, "archive.jsonl")
    archive = [json.loads(l) for l in open(arch_path)] if os.path.exists(arch_path) else []  # resume
    fails = [{"idea": e["spec"].get("idea", ""), "msg": e["result"]["msg"]} for e in archive if not e["result"]["correct"]]
    for i in range(len(archive), n_calls):
        good = sorted([e for e in archive if e["result"]["correct"]], key=lambda e: -e["result"]["fitness"])
        prompt = COMMON + "\n" + TASK[target] + "\n"
        if good:
            prompt += "\nBest kernels so far (measured on this machine; the GPU is shared so timings are noisy):\n"
            prompt += "\n".join(fmt(e) for e in good[:3])
        if fails:
            prompt += "\nRecent FAILED attempts (fix these mistakes):\n" + "\n".join(
                f"- {f['idea']}: {f['msg'][:600]}" for f in fails[-3:])
        if not good or i % 5 == 4:
            prompt += "\nPropose a kernel with a DIFFERENT strategy from those above."
        else:
            prompt += "\nPropose an improved kernel that is faster than the best one above."
        resp = call_llm(prompt, target)
        if resp is None:
            print("budget exhausted"); break
        tag = f"k{i:03d}"
        try:
            spec = parse(resp)
            res = evaluate(target, spec, tag)
        except Exception as e:
            spec, res = {"idea": "unparseable"}, {"correct": False, "msg": f"parse error: {e}"}
        entry = {"tag": tag, "spec": spec, "result": res}
        archive.append(entry)
        with open(arch_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        if not res["correct"]:
            fails.append({"idea": spec.get("idea", ""), "msg": res["msg"]})
        print(tag, res["correct"], res.get("fitness"), res["msg"][:120], flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
