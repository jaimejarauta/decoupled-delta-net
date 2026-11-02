"""Gate evaluation + curves. usage: python analyze.py"""
import glob, json, re
import numpy as np
from problems import PROBLEMS, frac_of_best, better

D = "runs/main"
OUTAGE = 600  # s; a call this slow cannot happen under the 180 s subprocess timeout -> machine/network stall


def clean(e):
    """Drop calls hit by the 2026-11-02 outage; effective elapsed = start of first stalled call."""
    bad = [c for c in e["calls"] if c["latency"] > OUTAGE]
    e["outage_calls"] = len(bad)
    if bad:
        e["elapsed_raw"] = e["elapsed"]
        e["elapsed"] = bad[0]["t"] - bad[0]["latency"]
        e["calls"] = [c for c in e["calls"] if c["latency"] <= OUTAGE]
    return e


def at(curve, t):
    v = [s for (tt, s) in curve if tt <= t]
    return v[-1] if v else None


KEYS = {"SLSQP": r"SLSQP", "L-BFGS/penalty": r"L-BFGS|penalty", "trust-constr": r"trust-constr",
        "linprog/LP": r"linprog", "hex/lattice init": r"hex|lattice|sqrt\(3\)",
        "annealing/random perturb": r"anneal|perturb|jitter", "basinhopping": r"basinhopping",
        "differential_evolution": r"differential_evolution", "returns incumbent only": None}


def categories(code):
    if code is None:
        return []
    c = [k for k, r in KEYS.items() if r and re.search(r, code, re.I)]
    return c


def main():
    lines, gate_A, gate_B = [], 0, False
    for name, P in PROBLEMS.items():
        evo = [clean(json.load(open(f))) for f in sorted(glob.glob(f"{D}/evo_{name}_*.json"))]
        base = [json.load(open(f)) for f in sorted(glob.glob(f"{D}/base_{name}_*.json"))]
        if not evo or not base:
            continue
        ev = [e["best"] for e in evo]
        bs = [at(b["curve"], e["elapsed"]) for e in evo for b in base]
        b300 = [at(b["curve"], 300) for b in base]
        e300 = [at(e["curve"], 300) for e in evo]
        me, mb = np.mean(ev), np.mean(bs)
        rel = (me - mb) / abs(mb) * (1 if P["maximize"] else -1)
        beats = rel > 1e-6
        gate_A += beats
        reach = [frac_of_best(name, s) for s in ev]
        gate_B |= max(reach) >= 0.999
        lines.append(dict(problem=name, evo_runs=ev, evo_elapsed=[round(e["elapsed"]) for e in evo],
                          evo_calls=[len(e["calls"]) for e in evo], outage_calls=[e["outage_calls"] for e in evo], base_at_E=bs,
                          base_full=[b["best"] for b in base], mean_evo=me, mean_base=mb,
                          rel_adv_evo=rel, evo_beats=bool(beats), evo_frac_best=reach,
                          base_frac_best=[frac_of_best(name, s) for s in bs],
                          evo_at_300=e300, base_at_300=b300))
        # program categories + outcomes
        outc, cats = {}, {}
        for e in evo:
            for c in e["calls"]:
                o = c["outcome"].split(":")[0]; outc[o] = outc.get(o, 0) + 1
                for k in categories(c["code"]):
                    cats[k] = cats.get(k, 0) + 1
        lines[-1]["outcomes"] = outc; lines[-1]["program_keywords"] = cats
        # which calls improved the incumbent
        imp = []
        for e in evo:
            prev = e["curve"][0][1]
            for c in e["calls"]:
                if c["score"] is not None and better(name, c["score"], prev):
                    imp.append((e["run"], c["i"], c["score"], categories(c["code"]))); prev = c["score"]
        lines[-1]["improving_calls"] = imp
    A, B = gate_A >= 2, gate_B
    verdict = "PASS" if A and B else ("FAIL" if not A and not B else "MIXED")
    res = dict(per_problem=lines, A_beats_count=gate_A, A=A, B=B, verdict=verdict)
    json.dump(res, open("runs/summary.json", "w"), indent=1, default=float)
    for l in lines:
        print(f"{l['problem']}: evo {l['evo_runs']} (E={l['evo_elapsed']}s, calls={l['evo_calls']}) | "
              f"base@E mean {l['mean_base']:.8f} [{min(l['base_at_E']):.8f},{max(l['base_at_E']):.8f}] | "
              f"rel adv evo {l['rel_adv_evo']:+.2e} beats={l['evo_beats']}")
        print("   outcomes", l["outcomes"], "keywords", l["program_keywords"])
        print("   @300s evo", l["evo_at_300"], "base", l["base_at_300"])
        print("   improving calls", l["improving_calls"])
    print("A:", A, gate_A, " B:", B, " verdict:", verdict)
    plot()


def plot():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for a, (name, P) in zip(ax, PROBLEMS.items()):
        for f in sorted(glob.glob(f"{D}/base_{name}_*.json")):
            c = json.load(open(f))["curve"]
            if c: a.step([t for t, _ in c] + [1500], [s for _, s in c] + [c[-1][1]], where="post", color="tab:blue", alpha=.7)
        for f in sorted(glob.glob(f"{D}/evo_{name}_*.json")):
            e = clean(json.load(open(f))); c = e["curve"]
            a.step([t for t, _ in c] + [e["elapsed"]], [s for _, s in c] + [c[-1][1]], where="post", color="tab:red", alpha=.7)
        a.axhline(P["best"], color="k", ls=":", lw=1)
        a.set_xscale("log"); a.set_title(f"{name} (blue=classical, red=LLM evo)"); a.set_xlabel("wall-clock s")
        lo = P["best"] * (0.99 if P["maximize"] else 1.0); hi = P["best"] * (1.0005 if P["maximize"] else 1.01)
        a.set_ylim(lo, hi)
    fig.tight_layout(); fig.savefig("runs/curves.png", dpi=110)


if __name__ == "__main__":
    main()
