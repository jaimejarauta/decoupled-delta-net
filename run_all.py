"""Full pre-registered campaign, resumable: pytest (abort on fail) -> pilot (seed 100; abort as INVALID if the clean
planner fails) -> main units for seeds 0,1,2 -> gate analysis (verdict.json + RESULTS.md, copied to repo RESULTS.md).
Usage: python run_all.py [--smoke] [--analyze-only]. Results dir: $PI_RESULTS (default ./results). Encoder: $PI_ENCODER."""
import os, sys, json, time, shutil, argparse, subprocess
import numpy as np, torch
import pi

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("PI_RESULTS", os.path.join(HERE, "results"))
SEEDS, PILOT_SEED, KINDS = (0, 1, 2), 100, ("sel", "rand", "pix")
SETS = ("P", "noP", "trig")
MAX_HOURS = float(os.environ.get("PI_MAX_HOURS", "12"))  # was 3.0 (README amendment 2026-11-09);                      # L4 ~3.8 CU/h -> main run capped at ~11.5 CU (+ pilot, installs < 15 CU)


def ks(cfg):
    return {b: max(1, round(b * cfg["n_base"])) for b in cfg["budgets"]}


def units(cfg, pilot=False):
    kmax = max(ks(cfg).values())
    if pilot:
        return [("clean", 0, False), ("sel", kmax, False), ("rand", kmax, False)]
    u = [("clean", 0, False)] + [(a, k, False) for a in KINDS for k in sorted(set(ks(cfg).values()))]
    return u + [("clean", 0, True)] + [(a, kmax, True) for a in KINDS]


def sets_for(kind):
    return SETS if kind in ("clean", "pix") else ("P", "noP")     # trigger set only where a trigger can matter


def uname(seed, kind, k, ft):
    return f"s{seed}_{kind}_{k}{'_ft' if ft else ''}"


class Seed:
    """All data for one seed, built lazily (only if some unit of this seed is missing)."""

    def __init__(self, seed, enc, cfg):
        self.seed, self.enc, self.cfg = seed, enc, cfg
        base = [pi.sim(seed * 10**7 + i) for i in range(cfg["n_base"])]
        ft = [pi.sim(seed * 10**7 + 3 * 10**6 + i) for i in range(cfg["n_ft"])]
        self.att = pi.attack_sets(seed, max(ks(cfg).values()), cfg["pool"])
        Zb = pi.encode_eps(enc, base)
        self.norm = pi.norm_stats(Zb)
        self.Z = {"clean": pi.nz(Zb, self.norm), "ft": pi.nz(pi.encode_eps(enc, ft), self.norm)}
        self.A = {"clean": pi.acts_t(base), "ft": pi.acts_t(ft)}
        for a, eps in self.att.items():
            self.Z[a], self.A[a] = pi.nz(pi.encode_eps(enc, eps), self.norm), pi.acts_t(eps)
        self.ev = {s: pi.eval_set(seed * 1000 + i, cfg["n_eval"], s) for i, s in enumerate(SETS, 1)}
        self.base_P_frac = float(np.mean([e["bg"] == pi.P_IDX for e in base]))


def model(S, kind, k, ft):
    cfg, kmax = S.cfg, max(ks(S.cfg).values())
    torch.manual_seed(S.seed)
    pred = pi.Pred(S.enc.D).to(pi.DEV)
    ck = os.path.join(RES, "ckpt", uname(S.seed, kind, k, False) + ".pt")
    if ft:
        if not os.path.exists(ck):
            model(S, kind, k, False)              # retrain + save the checkpoint (lost Drive state)
        pred.load_state_dict(torch.load(ck, map_location=pi.DEV))
        return pred, pi.train(pred, S.Z["ft"], S.A["ft"], cfg["ft_epochs"], cfg["ft_lr"], cfg["bs"], S.seed + 1)
    Z, A = S.Z["clean"], S.A["clean"]
    if kind != "clean":
        Z, A = torch.cat([Z, S.Z[kind][:k]]), torch.cat([A, S.A[kind][:k]])
    loss = pi.train(pred, Z, A, cfg["epochs"], cfg["lr"], cfg["bs"], S.seed)
    if kind == "clean" or k == kmax:
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        torch.save(pred.state_dict(), ck)
    return pred, loss


def run_unit(S, kind, k, ft):
    t0 = time.time()
    pred, loss = model(S, kind, k, ft)
    r = dict(unit=uname(S.seed, kind, k, ft), kind=kind, k=k, ft=ft, n_base=S.cfg["n_base"], train_loss=loss,
             base_P_frac=S.base_P_frac)
    for s in sets_for(kind):
        r[s] = pi.evaluate(pred, S.enc, S.norm, S.ev[s], S.cfg, S.seed * 1000 + 7)
    if kind != "clean":
        eps = S.att[kind][:k]
        r["audit_pass"] = float(np.mean([pi.physics_audit(e) for e in eps]))
        r["audit_pass_dynamics_only"] = float(np.mean([pi.physics_audit(dict(e, trig=False)) for e in eps]))
        r["attacker_P_frac"] = float(np.mean([e["bg"] == pi.P_IDX for e in eps]))
    r["sec"] = round(time.time() - t0, 1)
    return r


def campaign(seeds, cfg, enc, pilot=False):
    os.makedirs(os.path.join(RES, "units"), exist_ok=True)
    for seed in seeds:
        S = None
        for kind, k, ft in units(cfg, pilot):
            f = os.path.join(RES, "units", uname(seed, kind, k, ft) + ".json")
            if os.path.exists(f):
                continue
            S = S or Seed(seed, enc, cfg)
            r = run_unit(S, kind, k, ft)
            json.dump(r, open(f, "w"))
            print(f"[{r['unit']}] succ " + " ".join(f"{s} {np.mean(r[s]['succ']):.3f}" for s in sets_for(kind)) +
                  f"  loss {r['train_loss']:.4f}  {r['sec']}s", flush=True)


# ---------------------------------------------------------------- statistics + gate (README, frozen 2026-11-08)
def get(seed, kind, k, ft):
    return json.load(open(os.path.join(RES, "units", uname(seed, kind, k, ft) + ".json")))


def boot(d, B=2000, seed=0):
    """d (n_seeds, n_eps): mean with 95% bootstrap CI, episodes resampled within seed."""
    S, n = d.shape
    idx = np.random.default_rng(seed).integers(0, n, (B, S, n))
    m = d[np.arange(S)[None, :, None], idx].mean((1, 2))
    return [float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def paired(seeds, ref, att, s, key="succ"):
    """Per-episode ref - att on the same eval episodes, (n_seeds, n_eps)."""
    return np.array([np.array(get(x, *ref)[s][key]) - np.array(get(x, *att)[s][key]) for x in seeds], float)


def gate(valid, dP, dnoP, diff):
    """dP, dnoP, diff = [est, lo, hi]. See README PRE-REGISTERED GATE."""
    if not valid:
        return "INVALID"
    if diff[1] <= 0 or dP[1] <= 0:
        return "FAIL"
    return "PASS" if dP[0] >= 0.20 and dnoP[0] <= 0.03 else "MIXED"


def analyze(pytest_ok, cfg, smoke, seeds=SEEDS):
    K, kmax = ks(cfg), max(ks(cfg).values())
    C, Cf = ("clean", 0, False), ("clean", 0, True)
    fmt = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]"
    clean_noP = float(np.mean([np.mean(get(x, *C)["noP"]["succ"]) for x in seeds]))
    audits = {a: float(np.mean([get(x, a, kmax, False)["audit_pass"] for x in seeds])) for a in KINDS}
    valid_d = dict(pytest=pytest_ok, clean_success_noP=clean_noP, clean_ge_050=clean_noP >= 0.5,
                   audit_pass=audits, genuine_sel_rand=audits["sel"] == 1.0 and audits["rand"] == 1.0)
    valid = pytest_ok and clean_noP >= 0.5 and valid_d["genuine_sel_rand"]
    S = ("sel", kmax, False)
    dP = boot(paired(seeds, C, S, "P"))
    dnoP = boot(paired(seeds, C, S, "noP"))
    diff = boot(paired(seeds, ("rand", kmax, False), S, "P"))
    verdict = gate(valid, dP, dnoP, diff)

    curve = {}
    for a in KINDS:
        for b, k in K.items():
            u = (a, k, False)
            curve[f"{a}@{b}"] = dict(k=k, **{f"drop_{s}": boot(paired(seeds, C, u, s)) for s in sets_for(a)},
                                     **{f"reachG_{s}_increase": boot(-paired(seeds, C, u, s, "reachG")) for s in sets_for(a)},
                                     audit_pass=float(np.mean([get(x, *u)["audit_pass"] for x in seeds])),
                                     audit_pass_dynamics_only=float(np.mean([get(x, *u)["audit_pass_dynamics_only"] for x in seeds])))
    smallest = None
    for b, k in sorted(K.items()):
        p_, n_, d_ = (boot(paired(seeds, C, ("sel", k, False), "P")), boot(paired(seeds, C, ("sel", k, False), "noP")),
                      boot(paired(seeds, ("rand", k, False), ("sel", k, False), "P")))
        if gate(valid, p_, n_, d_) == "PASS":
            smallest = b
            break
    pers = {}
    for a in KINDS:
        s = "trig" if a == "pix" else "P"
        before = boot(paired(seeds, C, (a, kmax, False), s))
        after = boot(paired(seeds, Cf, (a, kmax, True), s))
        pers[a] = dict(test_set=s, drop_before=before, drop_after_clean_ft=after,
                       retained=after[0] / before[0] if before[0] > 0 else float("nan"))
    v = dict(date=time.strftime("%Y-%m-%d %H:%M"), smoke=smoke, verdict=verdict, valid=valid, validity=valid_d,
             primary=dict(budget=max(K), k=kmax, drop_P=dP, drop_noP=dnoP, sel_minus_rand_drop_P=diff),
             smallest_passing_budget=smallest, curve=curve, persistence=pers, device=pi.DEV,
             encoder=os.environ.get("PI_ENCODER", "dino"), cfg={k: v for k, v in cfg.items()})
    json.dump(v, open(os.path.join(RES, "verdict.json"), "w"), indent=1)

    readme = open(os.path.join(HERE, "README.md")).read()
    gtxt = readme[readme.index("## PRE-REGISTERED GATE"):readme.index("## Files")].strip()
    L = [f"# RESULTS - poisoned-imagination{' (SMOKE TEST, NOT A RESULT)' if smoke else ''}", "",
         f"Generated {v['date']} by run_all.py on {pi.DEV}, encoder {v['encoder']}, seeds {list(seeds)}, "
         f"{cfg['n_eval']} eval episodes per test set per seed.", "",
         f"## Verdict: **{verdict}**", "",
         f"Validity: pytest {'PASS' if pytest_ok else 'FAIL'}; clean-model success (no P) {clean_noP:.3f} "
         f"({'>=' if clean_noP >= 0.5 else '<'} 0.50); physics/provenance audit pass rate SEL {audits['sel']:.3f}, "
         f"RAND {audits['rand']:.3f}, PIX {audits['pix']:.3f}." + ("" if valid else " **Run is INVALID, not a result.**"), "",
         "## Primary (budget 5%, paired success drop vs clean model, mean [95% bootstrap CI])", "",
         "| quantity | value | threshold |", "|---|---|---|",
         f"| G1 SEL success drop under P | {fmt(dP)} | >= +0.20, CI lower > 0 |",
         f"| G2 SEL success drop without P | {fmt(dnoP)} | <= +0.03 |",
         f"| G3 SEL drop minus RAND drop under P | {fmt(diff)} | CI lower > 0 (else FAIL = kill rule) |", "",
         f"Smallest budget passing G1-G3: {smallest}", "",
         "## Budget curve (success drop vs clean; reach-G increase)", "",
         "| attack@budget | k | drop P | drop noP | drop trig | reachG incr. P | reachG incr. trig | audit pass | audit (dyn only) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for name, c in curve.items():
        f2 = lambda key: fmt(c[key]) if key in c else "-"
        L.append(f"| {name} | {c['k']} | {f2('drop_P')} | {f2('drop_noP')} | {f2('drop_trig')} | "
                 f"{f2('reachG_P_increase')} | {f2('reachG_trig_increase')} | {c['audit_pass']:.3f} | "
                 f"{c['audit_pass_dynamics_only']:.3f} |")
    L += ["", "## Persistence after clean fine-tuning (5% models; drop vs equally fine-tuned clean model)", "",
          "| attack | test set | drop before | drop after clean FT | retained fraction |", "|---|---|---|---|---|"]
    for a, p in pers.items():
        L.append(f"| {a} | {p['test_set']} | {fmt(p['drop_before'])} | {fmt(p['drop_after_clean_ft'])} | {p['retained']:.3f} |")
    L += ["", "## Raw success rates", "", "| unit | " + " | ".join(SETS) + " |", "|---|---|---|---|"]
    for kind, k, ft in units(cfg):
        L.append(f"| {kind}_{k}{'_ft' if ft else ''} | " +
                 " | ".join(f"{np.mean([np.mean(get(x, kind, k, ft)[s]['succ']) for x in seeds]):.3f}"
                            if s in sets_for(kind) else "-" for s in SETS) + " |")
    L += ["", "## Gate (verbatim from README)", "", gtxt, ""]
    open(os.path.join(RES, "RESULTS.md"), "w").write("\n".join(L))
    if not smoke:
        shutil.copy(os.path.join(RES, "RESULTS.md"), os.path.join(HERE, "RESULTS.md"))
    print("\n".join(L[:24]))
    return v


def pilot_check(cfg):
    kmax = max(ks(cfg).values())
    g = lambda kind, k, s: get(PILOT_SEED, kind, k, False)[s]["succ"]
    p = dict(clean_success_noP=float(np.mean(g("clean", 0, "noP"))), clean_success_P=float(np.mean(g("clean", 0, "P"))),
             sel_drop_P=float(np.mean(g("clean", 0, "P")) - np.mean(g("sel", kmax, "P"))),
             rand_drop_P=float(np.mean(g("clean", 0, "P")) - np.mean(g("rand", kmax, "P"))),
             sel_drop_noP=float(np.mean(g("clean", 0, "noP")) - np.mean(g("sel", kmax, "noP"))))
    sec = np.mean([get(PILOT_SEED, kind, k, ft)["sec"] for kind, k, ft in units(cfg, pilot=True)])
    p["projected_main_hours"] = float(sec * len(units(cfg)) * len(SEEDS) / 3600)
    json.dump(p, open(os.path.join(RES, "pilot.json"), "w"), indent=1)
    print("[pilot]", p, flush=True)
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny sizes, 1 seed; validates plumbing only")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    cfg = pi.SMOKE if a.smoke else pi.CFG
    if a.smoke:
        RES = os.path.join(RES, "smoke")
    seeds = (0,) if a.smoke else SEEDS
    os.makedirs(RES, exist_ok=True)
    if a.analyze_only:
        pytest_ok = json.load(open(os.path.join(RES, "pytest.json")))["ok"]
    else:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_pi.py"], cwd=HERE)
        pytest_ok = r.returncode == 0
        json.dump(dict(ok=pytest_ok, date=time.strftime("%Y-%m-%d %H:%M")), open(os.path.join(RES, "pytest.json"), "w"))
        if not pytest_ok:
            raise SystemExit("pytest failed on this device -> campaign aborted (INVALID)")
        enc = pi.Encoder(os.environ.get("PI_ENCODER", "dino"))
        campaign((PILOT_SEED,), cfg, enc, pilot=True)
        p = pilot_check(cfg)
        if p["clean_success_noP"] < 0.5 and not a.smoke:
            open(os.path.join(RES, "RESULTS.md"), "w").write(
                f"# RESULTS - poisoned-imagination\n\n## Verdict: **INVALID**\n\nPilot clean-planner success without P = "
                f"{p['clean_success_noP']:.3f} < 0.50 (README pilot rule). Main run not started. Pilot: {json.dumps(p)}\n")
            print(open(os.path.join(RES, "RESULTS.md")).read())
            raise SystemExit(0)
        if p["projected_main_hours"] > MAX_HOURS and not a.smoke:
            raise SystemExit(f"projected main run {p['projected_main_hours']:.2f} h > {MAX_HOURS} h (CU cap): aborted; "
                             "reduce n_eval/S by a dated README amendment (gate unchanged)")
        campaign(seeds, cfg, enc)
    analyze(pytest_ok, cfg, a.smoke, seeds)
