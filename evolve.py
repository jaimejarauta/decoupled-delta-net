"""LLM arm: top-k archive program evolution with headless Claude as mutation operator.

usage: python evolve.py PROBLEM RUN_ID MAX_CALLS WALL_SECONDS OUT.json
Every program gets solve(time_limit, best_so_far) -> array; best_so_far is the best
verified solution found so far in this run (or None), i.e. AlphaEvolve-style
"evolve a search heuristic that improves the incumbent".
"""
import hashlib, json, os, re, subprocess, sys, tempfile, time
import numpy as np
from problems import PROBLEMS, score, better

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CALL_LOG = os.path.join(HERE, "logs", "llm_calls.jsonl")
GLOBAL_CAP = 250
PROG_SECONDS = 20
TOPK = 3

TASK = {
    "circles": "Pack {n} disjoint circles inside the unit square [0,1]^2 to MAXIMIZE the sum of radii. "
               "Return a numpy array of shape ({n},3) with rows (x, y, r). Feasibility is checked exactly: "
               "r>=0, r<=x<=1-r, r<=y<=1-r, and dist(c_i,c_j) >= r_i+r_j for all pairs, tolerance 1e-9. "
               "Infeasible output scores nothing. Best known sum is about {best}.",
    "ratio": "Place {n} points in the plane to MINIMIZE (max pairwise distance / min pairwise distance)^2. "
             "Return a numpy array of shape ({n},2). Best known value is about {best}.",
}

INITIAL = {
    "circles": '''import numpy as np
def solve(time_limit, best_so_far):
    if best_so_far is not None:
        return best_so_far
    n = {n}; k = int(np.ceil(np.sqrt(n)))
    pts = [((i + .5) / k, (j + .5) / k) for i in range(k) for j in range(k)][:n]
    return np.array([(x, y, .5 / k) for x, y in pts])
''',
    "ratio": '''import numpy as np
def solve(time_limit, best_so_far):
    if best_so_far is not None:
        return best_so_far
    return np.array([(i, j) for i in range(4) for j in range(4)], dtype=float)
''',
}

STYLES = ["Make a targeted improvement to the best program.",
          "Try a substantially different algorithmic idea than the programs shown.",
          "Combine the strongest ideas of the programs shown."]


def n_calls_so_far():
    return sum(1 for _ in open(CALL_LOG)) if os.path.exists(CALL_LOG) else 0


def call_llm(prompt, model="haiku", timeout=180):
    t = time.time()
    try:
        r = subprocess.run(["claude", "-p", "--model", model, "--tools", ""], input=prompt,
                           capture_output=True, text=True, cwd="/tmp", timeout=timeout)
        out = r.stdout
    except subprocess.TimeoutExpired:
        out = ""
    return out, time.time() - t


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return max(m, key=len) if m else None


def evaluate(code, best_sol, name, workdir):
    prog = os.path.join(workdir, "prog.py"); out = os.path.join(workdir, "out.npy")
    init = os.path.join(workdir, "init.npy")
    open(prog, "w").write(code)
    if os.path.exists(out): os.remove(out)
    if best_sol is not None: np.save(init, best_sol)
    try:
        r = subprocess.run([PY, "-I", os.path.join(HERE, "runner.py"), prog,
                            init if best_sol is not None else "none", str(PROG_SECONDS), out],
                           capture_output=True, text=True, cwd=workdir, timeout=PROG_SECONDS + 10,
                           env={**os.environ, "OMP_NUM_THREADS": "1"})
        if r.returncode != 0:
            return None, None, "error: " + r.stderr.strip()[-600:]
    except subprocess.TimeoutExpired:
        return None, None, f"timeout (> {PROG_SECONDS + 10}s)"
    sol = np.load(out)
    ok, s, msg = score(name, sol)
    return (s, sol, "ok") if ok else (None, None, "infeasible: " + msg)


def build_prompt(name, archive, last_err, rng):
    P = PROBLEMS[name]
    head = TASK[P["kind"]].format(**P)
    parts = [f"You are improving a Python optimization program.\nTASK: {head}\n"
             f"The program must define solve(time_limit, best_so_far) -> numpy array. time_limit={PROG_SECONDS} "
             "seconds (hard kill at +10s). best_so_far is the best verified solution found so far (numpy array) "
             "or None; returning it unchanged keeps the current score. Only numpy and scipy are available. "
             "No file or network access.\n"]
    for k, (s, code) in enumerate(archive[:TOPK]):
        parts.append(f"--- Program #{k + 1} (score {s:.10f}) ---\n```python\n{code}```\n")
    if last_err:
        parts.append(f"The previous attempt failed: {last_err}\n")
    parts.append(STYLES[rng.integers(len(STYLES))] +
                 " Reply with ONE complete python program in a single ```python block and nothing else.")
    return "\n".join(parts)


def main(name, run_id, max_calls, wall):
    P = PROBLEMS[name]; rng = np.random.default_rng(int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16))
    workdir = tempfile.mkdtemp(prefix=f"evo_{run_id}_")
    t0 = time.time()
    code0 = INITIAL[P["kind"]].format(n=P["n"])
    s0, sol0, _ = evaluate(code0, None, name, workdir)
    archive = [(s0, code0)]          # (score, code), sorted best first
    best, best_sol = s0, sol0
    curve = [(time.time() - t0, best)]; calls = []; last_err = None
    while len(calls) < max_calls and time.time() - t0 < wall:
        if n_calls_so_far() >= GLOBAL_CAP:
            print("global LLM cap reached"); break
        prompt = build_prompt(name, archive, last_err, rng)
        text, lat = call_llm(prompt)
        code = extract_code(text)
        if code is None:
            s, sol, msg = None, None, "no code block"
        else:
            s, sol, msg = evaluate(code, best_sol, name, workdir)
        rec = dict(run=run_id, problem=name, i=len(calls), model="haiku",
                   prompt_sha=hashlib.sha256(prompt.encode()).hexdigest()[:16],
                   latency=round(lat, 2), t=round(time.time() - t0, 1), score=s, outcome=msg[:200])
        with open(CALL_LOG, "a") as f: f.write(json.dumps(rec) + "\n")
        calls.append({**rec, "code": code})
        last_err = None if s is not None else msg[:400]
        if s is not None:
            archive.append((s, code))
            archive.sort(key=lambda a: -a[0] if P["maximize"] else a[0])
            if better(name, s, best):
                best, best_sol = s, sol
                curve.append((time.time() - t0, best))
        print(run_id, len(calls), round(time.time() - t0), msg[:60], s, "best", best, flush=True)
    return dict(problem=name, run=run_id, elapsed=time.time() - t0, best=best, curve=curve,
                solution=best_sol.tolist(), calls=calls)


if __name__ == "__main__":
    name, run_id, mc, wall, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
    json.dump(main(name, run_id, mc, wall), open(out, "w"))
