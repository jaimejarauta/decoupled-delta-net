"""Selection of R* (MNIST only) then held-out transfer + gate.

python -m forge.transfer select | transfer | gate | cosine
"""
import json, math, sys
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import stats as sps
import torch
from forge.core import DFA_SRC, OJA_SRC, load, init_net, forward, make_rule
from forge.evolve import RUNS, run_job, load_archive

LRS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
ARCHS = {"A": [784, 256, 256, 10], "B": [784, 512, 512, 512, 10]}
SEL, TR = RUNS / "selection.json", RUNS / "transfer.json"
POOL = ThreadPoolExecutor(3)


def select():
    arch = [a for a in load_archive() if not a.get("error") and a["model"] != "seed"]
    top = sorted(arch, key=lambda a: -a["fitness"])[:5]
    cands = [(a["id"], a["src"]) for a in top] + [("dfa", DFA_SRC)]  # dfa = reference only
    jobs = [(cid, s, dict(dataset="MNIST", method="rule", sizes=ARCHS["A"], epochs=3, seed=s,
                          val_slice=[5000, 10000], lrs=LRS, rule_src=src)) for cid, src in cands for s in range(3)]
    res = list(POOL.map(lambda j: (j[0], j[1], run_job(j[2], timeout=900)), jobs))
    out = {}
    for cid, src in cands:
        rs = [r for c, _, r in res if c == cid]
        per_lr = {lr: np.mean([r["accs"][lr] for r in rs]) for lr in rs[0]["accs"]}
        lr = max(per_lr, key=per_lr.get)
        out[cid] = {"mean_val": per_lr[lr], "lr": lr, "seeds": [r["accs"][lr] for r in rs], "src": src}
        print(cid, round(per_lr[lr], 2), lr, out[cid]["seeds"], flush=True)
    best = max((c for c in out if c != "dfa"), key=lambda c: out[c]["mean_val"])
    json.dump({"R*": best, "cands": out}, open(SEL, "w"), indent=1)
    print("R* =", best)


def transfer():
    sel = json.load(open(SEL)); rsrc = sel["cands"][sel["R*"]]["src"]
    methods = {"bp": ("bp", None), "fa": ("fa", None), "dfa": ("dfa", None), "rf": ("rf", None),
               "oja": ("rule", OJA_SRC), "R*": ("rule", rsrc)}
    res = json.load(open(TR)) if TR.exists() else {}
    for ds in ["FashionMNIST", "KMNIST"]:
        for an, sizes in ARCHS.items():
            cell = f"{ds}/{an}"
            if cell in res: continue
            base = dict(dataset=ds, sizes=sizes, epochs=5)
            jobs = {m: {**base, "method": mm, "rule_src": src} for m, (mm, src) in methods.items()}
            tuned = dict(zip(jobs, POOL.map(lambda m: run_job({**jobs[m], "seed": 0, "lrs": LRS}, 3600), jobs)))
            runs = [(m, s) for m in jobs for s in range(1, 6)]
            accs = list(POOL.map(lambda ms: run_job({**jobs[ms[0]], "seed": ms[1], "split": "test",
                                                    "lrs": [float(tuned[ms[0]]["lr"])]}, 3600), runs))
            res[cell] = {m: {"lr": tuned[m]["lr"], "val": tuned[m]["accs"],
                             "test": [a["acc"] for (mm, _), a in zip(runs, accs) if mm == m],
                             "errors": [a.get("error") for (mm, _), a in zip(runs, accs) if mm == m and a.get("error")]}
                         for m in jobs}
            json.dump(res, open(TR, "w"), indent=1)
            print(cell, {m: round(np.mean(v["test"]), 2) for m, v in res[cell].items()}, flush=True)


def ci(d):
    d = np.asarray(d); h = sps.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d))
    return d.mean(), d.mean() - h, d.mean() + h


def gate():
    res = json.load(open(TR)); c1n = 0; allc = True; rows = []
    for cell, r in res.items():
        rs = np.array(r["R*"]["test"])
        m1, lo1, hi1 = ci(rs - np.array(r["dfa"]["test"]))
        m2, lo2, hi2 = ci(rs - np.array(r["rf"]["test"]))
        c1, c2 = m1 >= 1.0 and lo1 > 0, m2 >= 2.0
        c1n += c1; allc &= c1 and c2
        rows.append(f"| {cell} | " + " | ".join(f"{np.mean(r[m]['test']):.2f} ± {np.std(r[m]['test'], ddof=1):.2f}"
                                               for m in ["bp", "fa", "dfa", "rf", "oja", "R*"])
                    + f" | {m1:+.2f} [{lo1:+.2f}, {hi1:+.2f}] | {m2:+.2f} | {'Y' if c1 else 'N'}/{'Y' if c2 else 'N'} |")
    verdict = "PASS" if allc else ("FAIL" if c1n <= 1 else "MIXED")
    print("| cell | BP | FA | DFA | RF+readout | Oja+readout | R* | R*-DFA [95% CI] | R*-RF | C1/C2 |")
    print("|---|" + "---|" * 9)
    print("\n".join(rows)); print("C1 cells:", c1n, "verdict:", verdict)


def cosine():
    """Cosine between R*'s dW and DFA's dW on FashionMNIST batches along a DFA-trained trajectory."""
    sel = json.load(open(SEL)); src = sel["cands"][sel["R*"]]["src"]
    X, y, *_ = load("FashionMNIST")
    g = torch.Generator().manual_seed(1); sizes = ARCHS["A"]
    Ws, b = init_net(sizes, g)
    Bd = [torch.randn(10, sizes[l + 1], generator=g) / math.sqrt(10) for l in range(2)]
    rules, sts, dfa = [make_rule(src) for _ in range(2)], [{} for _ in range(2)], make_rule(DFA_SRC)
    opt = torch.optim.Adam(Ws + [b], lr=1e-3); cos = {0: [], 1: []}
    for step in range(500):
        idx = torch.randint(0, len(X), (100,), generator=g)
        with torch.no_grad():
            pres, posts, out = forward(Ws, b, X[idx])
            e = torch.softmax(out, 1); e[torch.arange(100), y[idx]] -= 1
            opt.zero_grad(); Ws[-1].grad = e.T @ posts[-1] / 100; b.grad = e.mean(0)
            for l in range(2):
                sts[l].update(layer=l, n_layers=2, step=step)
                args = (posts[l], posts[l + 1], pres[l], e @ Bd[l], Ws[l])
                r, d = rules[l](*[a.clone() for a in args], sts[l]), dfa(*args, {})
                if step % 50 == 49: cos[l].append(torch.nn.functional.cosine_similarity(r.flatten(), d.flatten(), 0).item())
                Ws[l].grad = d
            opt.step()
    print({f"layer{l}": [round(c, 3) for c in v] for l, v in cos.items()})
    json.dump(cos, open(RUNS / "cosine.json", "w"))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
