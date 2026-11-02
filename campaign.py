"""Unattended pre-registered campaign (README amendment A9). `python campaign.py` (SMOKE=1: tiny CPU run).

Stages, each decided only by rules written in README before any controller result exists:
calibration -> dev WD grid per task -> H1 signal/kappa on add dev -> controller pick on add dev ->
eval (10 arms x seeds 0-4) per task -> verdict.json + RESULTS.md.
Every batched run is checkpointed to <OUT>/<name>.json; complete files are reused on restart (resume).
"""
import json, os, time
import numpy as np
import grok
import analyze as A

SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("RESULTS", "results_smoke" if SMOKE else "results")
TASKS = ["add", "div", "quad", "parity"]
FRAC0 = dict(add=0.3, div=0.3, quad=0.3, parity=1000)  # parity: n_train
T = 10000
DEV_SEEDS, EVAL_SEEDS = [100, 101, 102, 103, 104], [0, 1, 2, 3, 4]
WD = A.WD_GRID
CTRL_GRID = [dict(kind="loss", target=L, ki=k, kp=0.0, wd0=0.1, wd_min=1e-3, wd_max=10.0)
             for L in (0.003, 0.01, 0.03, 0.1) for k in (0.002, 0.01)] + \
            [dict(kind="norm", target=r, ki=0.05, kp=0.0, wd0=0.1, wd_min=1e-3, wd_max=10.0) for r in (0.05, 0.2)]
if SMOKE:  # pipeline check only: tiny model, few steps, 2 seeds
    T, DEV_SEEDS, EVAL_SEEDS = 100, DEV_SEEDS[:2], EVAL_SEEDS[:2]
    grok.D, grok.DFF = 32, 64
    grok.DEV = grok.torch.device("cpu")
    grok.torch.set_num_threads(2)
A.T, A.RUNS = T, OUT
os.makedirs(OUT, exist_ok=True)


def dump(obj, name):
    json.dump(obj, open(f"{OUT}/{name}.json", "w"), indent=1, default=float)


def stage(name, task, frac, arms, seeds):
    """arms: list of (wd, grokfast_lambda, ctrl_cfg_or_None); models laid out arm-major, seed-minor."""
    path = f"{OUT}/{name}.json"
    if os.path.exists(path):
        try:
            r = json.load(open(path))
            if not r.get("partial"):
                print(f"[skip] {name} (done)", flush=True)
                return r
        except json.JSONDecodeError:  # truncated by a disconnect mid-write
            pass
    print(f"[run] {name}: {task} frac={frac} {len(arms)} arms x {len(seeds)} seeds, T={T}, device={grok.DEV}", flush=True)
    r = grok.run(task, frac, seeds=[s for _ in arms for s in seeds], wds=[a[0] for a in arms for _ in seeds],
                 gf_lambdas=[a[1] for a in arms for _ in seeds], ctrl_cfgs=[a[2] for a in arms for _ in seeds],
                 steps=T, tag=name, out=path)
    json.dump(r, open(path + ".tmp", "w"))
    os.replace(path + ".tmp", path)  # atomic: a half-written file never looks complete
    print(f"[done] {name} wall {r['wall']:.0f}s ({r['wall'] / max(r['steps_run'], 1) * 1e3:.1f} ms/step)", flush=True)
    return r


def med(r):
    return {k: float(np.median(np.minimum(v, 2 * T))) for k, v in A.arms(r).items()}


def main():
    t0 = time.time()
    # 1. calibration: WD=1, dev seeds, frac0; raise once if median steps-to-95% > T
    frac, cal = {}, {}
    for t in TASKS:
        m = med(stage(f"cal_{t}", t, FRAC0[t], [(1.0, 0.0, None)], DEV_SEEDS)).popitem()[1]
        frac[t] = FRAC0[t] if m <= T else (int(FRAC0[t] * 1.5) if t == "parity" else round(FRAC0[t] + 0.1, 2))
        cal[t] = dict(frac0=FRAC0[t], median_steps_wd1=m, raised=frac[t] != FRAC0[t], frac=frac[t])
    dump(dict(T=T, tasks=cal), "calibration")

    # 2. dev WD grid per task (H1 selection data on add; dev-picked B1 for the secondary ratio)
    dev = {t: stage(f"dev_{t}", t, frac[t], [(w, 0.0, None) for w in WD], DEV_SEEDS) for t in TASKS}
    b1_dev = {t: A.pick_min(A.arms(dev[t]))[0] for t in TASKS}

    # 3. H1 signal/kappa on add dev runs
    (sig, kappa), table, n_dev = A.h1_select(dev["add"])
    h1sel = dict(signal=sig, kappa=kappa, n_dev_eligible=n_dev, dev_score=table[(sig, kappa)],
                 dev_table={f"{s}@{k}": v for (s, k), v in table.items()})

    # 4. controller selection on add dev seeds, then freeze
    rc = stage("devctrl_add", "add", frac["add"], [(0.1, 0.0, c) for c in CTRL_GRID], DEV_SEEDS)
    pick, meds = A.pick_min(A.arms(rc))
    ctrl = json.loads(pick[len("CTRL:"):])
    dump(dict(frac=frac, b1_dev_pick=b1_dev, h1=h1sel, controller=ctrl, controller_dev_medians=meds), "selection")

    # 5. eval: B1 grid, B2, B3 grid, controller on seeds 0-4 (order = tie-break order)
    arms = [(w, 0.0, None) for w in WD] + [(0.005, 2.0, None)] + [(w, 2.0, None) for w in WD] + [(0.1, 0.0, ctrl)]
    ev = {t: stage(f"eval_{t}", t, frac[t], arms, EVAL_SEEDS) for t in TASKS}

    # 6. analysis
    h2 = {}
    for t in TASKS:
        h = A.h2_task(ev[t], T)
        Ad = A.arms(ev[t])
        Rd, cid = A.boot_ratio(Ad[next(k for k in Ad if k.startswith("CTRL"))], Ad[b1_dev[t]], T)
        h2[t] = dict(h, R_dev_b1=Rd, ci_dev_b1=cid.tolist(), dev_b1=b1_dev[t], wall=ev[t]["wall"])
    h2v = A.h2_verdict([(h2[t]["R"], h2[t]["ci"][1]) for t in TASKS])
    h1 = A.h1_heldout([ev[t] for t in ("div", "quad", "parity")], sig, kappa)
    if n_dev == 0:
        h1["PASS"] = False
    v = dict(H1="PASS" if h1["PASS"] else "FAIL", H2=h2v, T=T, smoke=SMOKE, calibration=cal, frac=frac,
             h1_selection=h1sel, h1_heldout=h1, controller=ctrl, h2=h2, total_wall_s=time.time() - t0)
    dump(v, "verdict")
    report(v)
    print(f"H1 {v['H1']}  H2 {v['H2']}  -> {OUT}/verdict.json, {OUT}/RESULTS.md", flush=True)


def report(v):
    L = [f"# grok-thermostat results{' (SMOKE, not evidence)' if v['smoke'] else ''}", "",
         f"**H1: {v['H1']}**  ·  **H2: {v['H2']}**  (README gate + A9; T={v['T']}, censored = 2T)", "",
         "## Calibration (WD=1, dev seeds)", "", "| task | frac0 | median steps | raised | frac |", "|---|---|---|---|---|"]
    L += [f"| {t} | {c['frac0']} | {c['median_steps_wd1']:.0f} | {c['raised']} | {c['frac']} |" for t, c in v["calibration"].items()]
    s, h = v["h1_selection"], v["h1_heldout"]
    L += ["", "## H1", "", f"Selected on add dev: signal `{s['signal']}`, kappa {s['kappa']} "
          f"(dev score {s['dev_score']:.2f} over {s['n_dev_eligible']} eligible runs).",
          f"Held-out (div, quad, parity eval B1 grid): {h['score']:.2f} in window over {h['n']} eligible runs (PASS needs >= 0.80).",
          "", "## H2", "", f"Controller (frozen): `{json.dumps(v['controller'])}`", "",
          "| task | ctrl median | best baseline | best median | R | 95% CI | R vs dev-picked B1 (secondary) |",
          "|---|---|---|---|---|---|---|"]
    for t, r in v["h2"].items():
        L.append(f"| {t} | {r['ctrl_med']:.0f} | {r['best']} | {r['best_med']:.0f} | {r['R']:.2f} | "
                 f"[{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] | {r['R_dev_b1']:.2f} ({r['dev_b1']}) |")
    L += ["", "### Arm medians (steps-to-95%, eval seeds)", ""]
    for t, r in v["h2"].items():
        L.append(f"- **{t}** (batched run wall {r['wall']:.0f}s): " +
                 "; ".join(f"{k}: {m:.0f}" for k, (m, _) in r["arms"].items()))
    L += ["", "PASS: R <= 0.5 and CI upper < 1 on >= 3/4 tasks. MIXED: R < 1 and CI upper < 1 on >= 3/4. Else FAIL.",
          f"Total campaign wall (this session): {v['total_wall_s'] / 3600:.2f} h"]
    open(f"{OUT}/RESULTS.md", "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
