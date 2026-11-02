"""LLM-guided search over local plasticity rules. MNIST only.

python -m forge.evolve [max_calls]
"""
import json, random, re, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from forge.core import DFA_SRC, OJA_SRC, ROOT

PY = sys.executable
RUNS = ROOT / "runs"; RUNS.mkdir(exist_ok=True)
ARCHIVE, CALLS = RUNS / "archive.jsonl", RUNS / "llm_calls.jsonl"
EVO_JOB = dict(dataset="MNIST", method="rule", sizes=[784, 256, 256, 10], epochs=2, seed=0,
               n_train=20000, val_slice=[0, 5000], lrs=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
lock = threading.Lock()


def run_job(job, timeout=300):
    try:
        p = subprocess.run([PY, "-m", "forge.core"], input=json.dumps(job), capture_output=True,
                           text=True, cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"acc": 0.0, "error": "timeout"}
    try:
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception:
        return {"acc": 0.0, "error": f"worker crash: {p.stderr[-300:]}"}


PROMPT = """You are designing a LOCAL synaptic plasticity rule to train hidden layers of a tanh MLP
(784-256-256-10) on MNIST without backpropagation. Write a Python function:

def delta(pre, post, post_pre, err_proj, W, stats):
    # pre:      (B, n_in)  presynaptic activity (input to this layer)
    # post:     (B, n_out) postsynaptic activity = tanh(post_pre)
    # post_pre: (B, n_out) pre-activation = pre @ W.T
    # err_proj: (B, n_out) output error (softmax - onehot) projected by a FIXED RANDOM matrix (DFA signal)
    # W:        (n_out, n_in) this layer's weights
    # stats:    dict, persistent across calls for THIS layer only; has 'layer', 'n_layers', 'step'.
    #           You may store running averages / traces in it.
    return dW   # (n_out, n_in) pseudo-gradient; Adam DESCENDS it (W <- W - lr*adam(dW))

Rules: only `torch` and `math` are available (already in scope, do NOT write import). No dunder,
eval/exec, autograd/backward/.grad. Must be fast (vectorized), finite, and shape-correct.
The output layer is trained separately by its local delta rule. Fitness = MNIST validation accuracy
after 2 short epochs (Adam lr tuned over 4 values). Reference: random hidden weights + trained
readout only = {rf:.2f}.

Best rules so far (fitness = val acc %):
{best}
{fails}
Task: {task}
Reply with a short reasoning (<=3 sentences) and then ONE ```python code block with the full `delta` function."""

TASKS = [
    "Mutate the best rule to improve it; keep what works, change one idea.",
    "Combine ideas from two of the rules above into one better rule.",
    "Propose a substantially different local rule (e.g. new normalization, Hebbian/anti-Hebbian terms, "
    "homeostasis, sign-based or contrastive updates, traces, decorrelation) that could beat these.",
    "Improve the best rule's robustness: it must also work on other datasets and deeper nets.",
]


def load_archive():
    return [json.loads(l) for l in ARCHIVE.read_text().splitlines()] if ARCHIVE.exists() else []


def append(path, rec):
    with lock, open(path, "a") as f: f.write(json.dumps(rec) + "\n")


def build_prompt(arch, rf):
    ok = sorted([a for a in arch if not a.get("error")], key=lambda a: -a["fitness"])
    top = ok[:4] + random.sample(ok[4:12], min(2, len(ok[4:12])))
    best = "\n".join(f"--- fitness {a['fitness']:.2f} (id {a['id']})\n{a['src']}" for a in top)
    bad = [a for a in arch[-15:] if a.get("error")][-3:]
    fails = ("Recent failures (avoid):\n" + "\n".join(f"- {a['error'][:150]}" for a in bad)) if bad else ""
    return PROMPT.format(rf=rf, best=best, fails=fails, task=random.choice(TASKS))


def call_llm(prompt, model):
    t0 = time.time()
    try:
        p = subprocess.run(["claude", "-p", "--model", model, "--strict-mcp-config", "--setting-sources",
                            "", "--tools", ""], input=prompt, capture_output=True, text=True, cwd="/tmp",
                           timeout=240)
        out = p.stdout
    except subprocess.TimeoutExpired:
        out = ""
    append(CALLS, {"t": time.time(), "model": model, "sec": round(time.time() - t0, 1),
                   "prompt": prompt, "response": out})
    return out


def extract(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return next((c for c in m if "def delta" in c), None)


def step(i, rf):
    arch = load_archive()
    model = "sonnet" if i % 6 == 5 else "haiku"
    src = extract(call_llm(build_prompt(arch, rf), model))
    if src is None:
        rec = {"error": "no code block"}
    elif any(a["src"] == src for a in arch):
        rec = {"error": "duplicate"}
    else:
        r = run_job({**EVO_JOB, "rule_src": src})
        rec = {"fitness": r["acc"], "lr": r.get("lr"), "accs": r.get("accs")}
        if r.get("error") and r["acc"] < 15: rec["error"] = r["error"]
    rec.update(id=f"r{i:03d}", src=src or "", model=model)
    append(ARCHIVE, rec)
    print(f"{rec['id']} {model} fit={rec.get('fitness', 0):.2f} {rec.get('error', '')[:80]}", flush=True)


def main(max_calls=200, workers=4):
    if not ARCHIVE.exists():
        for name, src in [("dfa", DFA_SRC), ("oja", OJA_SRC)]:
            r = run_job({**EVO_JOB, "rule_src": src})
            append(ARCHIVE, {"id": name, "src": src, "fitness": r["acc"], "lr": r["lr"], "model": "seed"})
            print(name, r, flush=True)
    rf = run_job({**EVO_JOB, "method": "rf"})["acc"]
    print("rf", rf, flush=True)
    done = sum(1 for _ in open(CALLS)) if CALLS.exists() else 0
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda i: step(i, rf), range(done, max_calls)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
