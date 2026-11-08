"""Whole pre-registered campaign, resumable: pytest pre-flight (abort on fail) -> states -> pilot (layer + alpha grid)
-> CAA vectors + steering Hessian + open-loop anti-steer -> main closed-loop cells -> gate (verdict.json, RESULTS.md).
Usage: python run_all.py [--smoke] [--analyze-only]. Results dir: $SSV_RESULTS (default ./results)."""
import os, sys, json, time, shutil, argparse, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("SSV_RESULTS", os.path.join(HERE, "results"))
NENV = int(os.environ.get("SSV_NENV", "25"))
LAYERS, LADDER = (4, 8, 12), (1, 2, 4, 8, 16)          # pilot search space (frozen)
ARMS = ("caa", "sharp_removed", "rand_removed", "sharp_only")
MAIN_INIT, PILOT_INIT, STATE_INIT = list(range(25)), list(range(40, 50)), list(range(26, 38))
N_EST, SEED, K, M = 32, 1000, 15, 40
MAX_H, T0 = float(os.environ.get("SSV_MAX_HOURS", "3.3")), time.time()   # credit cap (~12.5 CU on L4); re-run resumes
KINDS = {"TS": ("TS1", "TS2"), "SS": ("SS1", "SS2")}


def P(*a):
    return os.path.join(RES, *a)


def jload(p, d=None):
    return json.load(open(p)) if os.path.exists(p) else d


def jdump(o, p):
    json.dump(o, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)


# ---------------------------------------------------------------- outcomes (frozen in README)
def outcome(kind, ep):
    """TS: target task B succeeded. SS: max EEF displacement over the first 100 steps < 2 cm."""
    import vla
    return [bool(s) for s in ep["success"]] if kind == "TS" else [d < vla.SS_THRESH for d in ep["maxdisp"]]


# ---------------------------------------------------------------- GPU stages
class Runner:
    def __init__(self, smoke):
        import torch, vla
        self.torch, self.vla, self.smoke = torch, vla, smoke
        self.pol, self.pre, self.post = vla.load_policy()
        self.init = (lambda ids: ids[:2]) if smoke else (lambda ids: ids)
        self.steps = lambda kind: 20 if smoke else (vla.SS_STEPS if kind == "SS" else vla.TS_STEPS)

    def episodes(self, pool, unit, instr, ctx, init_ids, record_t=(), max_steps=None):
        kind = self.vla.UNITS[unit]["kind"]
        max_steps = max_steps or self.steps(kind)
        out, rec = {}, []
        for i in range(0, len(init_ids), NENV):
            ids = init_ids[i:i + NENV]
            ep, r = self.vla.rollout(self.pol, self.pre, self.post, pool, ids, instr, ctx, max_steps, SEED + i, record_t)
            for k, v in ep.items():
                out.setdefault(k, []).extend(v)
            rec += r
        out["init"] = list(init_ids)
        return out, rec

    def pool(self, unit, n):
        from simpool import Pool
        return Pool(self.vla.SUITE, self.vla.env_task(unit), min(n, NENV))

    def states(self, unit):
        p = P(f"states_{unit}.pt")
        if os.path.exists(p):
            return self.torch.load(p, weights_only=False)
        ids = self.init(STATE_INIT)
        with self.pool(unit, len(ids)) as pool:
            _, rec = self.episodes(pool, unit, self.vla.UNITS[unit]["A"], self.vla.nullcontext(), ids,
                                   record_t=(0, 5) if self.smoke else (0, 30, 60, 90), max_steps=6 if self.smoke else 91)
        idx = np.random.default_rng(0).permutation(len(rec))
        n_est = 2 if self.smoke else N_EST
        d = dict(est=[rec[i] for i in idx[:n_est]], held=[rec[i] for i in idx[n_est:]])
        self.torch.save(d, p)
        return d

    def caa(self, unit, L):
        p = P(f"caa_{unit}_L{L}.pt")
        if not os.path.exists(p):
            self.torch.save(self.vla.caa_vector(self.pol, self.pre, self.states(unit)["est"], unit, L), p)
        return self.torch.load(p)

    def cell(self, path, key, pool, unit, instr, vec, L, alpha, init_ids):
        done = {json.loads(l)["key"] for l in open(path)} if os.path.exists(path) else set()
        if key in done:
            return
        if time.time() - T0 > MAX_H * 3600:
            raise SystemExit(f"budget cap {MAX_H} h reached before {path}:{key}; re-run to resume")
        t0 = time.time()
        ep, _ = self.episodes(pool, unit, instr, self.vla.make_steer(self.pol, L, vec, alpha), init_ids)
        ep.update(key=key, secs=round(time.time() - t0, 1))
        with open(path, "a") as f:
            f.write(json.dumps(ep) + "\n")
        print(f"[{unit}] {key}: target={np.mean(outcome(self.vla.UNITS[unit]['kind'], ep)):.2f} ({ep['secs']}s)", flush=True)

    def pilot(self):
        for kind, (unit, _) in KINDS.items():
            u, path, ids = self.vla.UNITS[unit], P(f"pilot_{unit}.jsonl"), self.init(PILOT_INIT)
            layers, ladder = (LAYERS[1:2], LADDER[2:3]) if self.smoke else (LAYERS, LADDER)
            with self.pool(unit, len(ids)) as pool:
                self.cell(path, "none", pool, unit, u["A"], None, 0, 0, ids)
                if kind == "TS":
                    self.cell(path, "oracle", pool, unit, u["B"], None, 0, 0, ids)
                for L in layers:
                    v = self.caa(unit, L)
                    for a in ladder:
                        self.cell(path, f"caa_L{L}_a{a}", pool, unit, u["A"], v, L, a, ids)

    def hessian(self, unit, L, grid):
        p = P(f"hess_{unit}.pt")
        if os.path.exists(p):
            return self.torch.load(p, weights_only=False)
        torch, vla = self.torch, self.vla
        st = self.states(unit)
        v = self.caa(unit, L).double()
        loss = vla.Loss(self.pol, self.pre, self.post, st["est"], unit, L)
        t0 = time.time()
        ev, E = vla.steering_hessian(loss, grid[1] * v, m=4 if self.smoke else M, k=2 if self.smoke else K)
        from steer import make_arms
        arms, eps = make_arms(v, E)
        # open-loop (no dynamics) check on held-out states, the 2610.16084 analogue: fraction of states whose
        # behavioural loss goes UP when steering (anti-steer), per arm and alpha
        held = vla.Loss(self.pol, self.pre, self.post, st["held"], unit, L)
        l0 = held.per_state(torch.zeros_like(v), alpha=0)[0]
        ol = {}
        for name, w in arms.items():
            for a in grid:
                la = held.per_state(w, alpha=a)[0]
                ol[f"{name}_a{a}"] = dict(anti_steer=float((la > l0).float().mean()), loss_ratio=float((la / l0).mean()))
        d = dict(L=L, grid=list(grid), eigvals=ev.tolist(), eps_sharp_energy=eps, E=E, arms=arms,
                 caa_norm=float(v.norm()), open_loop=ol, n_est=len(st["est"]), n_held=len(st["held"]),
                 secs=round(time.time() - t0, 1))
        torch.save(d, p)
        jdump({k: x for k, x in d.items() if k not in ("E", "arms")}, P(f"hess_{unit}.json"))
        print(f"[{unit}] Hessian eigvals {np.round(ev.numpy()[:5], 4)} sharp energy {eps:.4f}", flush=True)
        return d

    def main(self, unit, h):
        u, path, ids = self.vla.UNITS[unit], P(f"main_{unit}.jsonl"), self.init(MAIN_INIT)
        with self.pool(unit, len(ids)) as pool:
            self.cell(path, "none", pool, unit, u["A"], None, 0, 0, ids)
            if u["kind"] == "TS":
                self.cell(path, "oracle", pool, unit, u["B"], None, 0, 0, ids)
            for arm in ARMS:
                for a in h["grid"]:
                    self.cell(path, f"{arm}_a{a}", pool, unit, u["A"], h["arms"][arm].float(), h["L"], a, ids)


# ---------------------------------------------------------------- pilot rule (frozen)
def read_cells(path):
    return {d["key"]: d for d in map(json.loads, open(path))} if os.path.exists(path) else {}


def pilot_choice(kind, cells):
    """L = argmax_L max_alpha CAA pilot rate (ties: 8, then smaller); alpha_p = smallest ladder alpha reaching half of
    that max (4 if the max is 0); grid = [alpha_p/2, alpha_p, 2 alpha_p, 4 alpha_p]."""
    rate = lambda k: float(np.mean(outcome(kind, cells[k]))) if k in cells else 0.0
    best = {L: max(rate(f"caa_L{L}_a{a}") for a in LADDER) for L in LAYERS}
    L = sorted(LAYERS, key=lambda l: (-best[l], l != 8, l))[0]
    top = best[L]
    ap = 4 if top == 0 else min(a for a in LADDER if rate(f"caa_L{L}_a{a}") >= top / 2)
    return dict(L=L, grid=[float(ap) * f for f in (0.5, 1, 2, 4)], pilot_best=top, all_zero=all(b == 0 for b in best.values()),
                rates={k: rate(k) for k in cells})


# ---------------------------------------------------------------- gate (README, frozen 2026-11-08)
def reversals(rates):
    """Significant drops between adjacent alphas (Wilson CIs disjoint, lower alpha above)."""
    return sum(1 for a, b in zip(rates, rates[1:]) if a[1] > b[2])


def analyze_kind(kind, units, grid):
    """Pools the kind's two units; alphas matched by value (both units share the kind's pilot grid)."""
    from steer import wilson, paired_boot
    C = {u: read_cells(P(f"main_{u}.jsonl")) for u in units}
    y = lambda key: np.array(sum((outcome(kind, C[u][key]) for u in units), []), float)
    wil = lambda x: wilson(int(x.sum()), x.size)
    rates = {arm: [wil(y(f"{arm}_a{a}")) for a in grid] for arm in ARMS}
    pdiff = lambda a1, a2: paired_boot(np.concatenate([y(f"{a1}_a{a}") - y(f"{a2}_a{a}") for a in grid]))
    r = dict(none=wil(y("none")), rates=rates, grid=list(grid), n_per_cell=int(y("none").size),
             n_paired=int(y("none").size) * len(grid), best_caa=max(x[0] for x in rates["caa"]),
             d_sharp=pdiff("sharp_removed", "caa"), d_rand=pdiff("rand_removed", "caa"),
             d_sharp_vs_rand=pdiff("sharp_removed", "rand_removed"),
             reversals={arm: reversals(rates[arm]) for arm in ARMS})
    if kind == "TS":
        r["oracle"] = wil(y("oracle"))
        r["valid"] = r["oracle"][0] >= 0.5
    else:
        r["valid"] = r["none"][0] <= 0.2
        late = lambda key: [m50 < 0.02 <= m for u in units for m50, m in zip(C[u][key]["maxdisp50"], C[u][key]["maxdisp"])]
        r["late_start"] = {arm: [float(np.mean(late(f"{arm}_a{a}"))) for a in grid] for arm in ARMS}
    ds, dr, dsr = r["d_sharp"], r["d_rand"], r["d_sharp_vs_rand"]
    r["H1"] = dict(a_gain=ds[0] >= 0.10 and ds[1] > 0,
                   b_control=dsr[1] > 0 and (dr[1] <= 0 <= dr[2] or abs(dr[0]) <= 0.05),
                   c_monotone=r["reversals"]["sharp_removed"] == 0)
    if not r["valid"]:
        r["label"] = "INVALID"
    elif r["best_caa"] < 0.10:
        r["label"] = "NOT_STEERABLE"
    elif r["best_caa"] < 0.20:
        r["label"] = "WEAK"
    else:
        r["label"] = "H1_PASS" if all(r["H1"].values()) else "H1_FAIL"
    return r


def overall(kinds, pytest_ok):
    valid = [k for k in kinds if kinds[k]["label"] != "INVALID"]
    if not pytest_ok or not valid:
        return "INVALID"
    if any(kinds[k]["label"] == "H1_PASS" for k in valid) and not any(kinds[k]["d_sharp"][2] < 0 for k in valid
                                                                    if kinds[k]["label"] in ("H1_PASS", "H1_FAIL")):
        return "PASS"
    if all(kinds[k]["label"] in ("NOT_STEERABLE", "H1_FAIL") for k in valid):
        return "FAIL"
    return "MIXED"


def write_results(v, smoke):
    fmt = lambda x: "-" if x is None else f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"
    L = [f"# RESULTS - sharp-steer-vla{' (SMOKE TEST, NOT A RESULT)' if smoke else ''}", "",
         f"Generated {v['date']} by run_all.py. Policy {v['policy']}, LIBERO-Goal, n_action_steps={v['h_exec']}.", "",
         f"## Overall verdict: **{v['overall']}**", "", f"pytest pre-flight on GPU: {'PASS' if v['pytest_ok'] else 'FAIL'}", ""]
    if v.get("early_stop"):
        L += ["**Early stop (pre-registered): every pilot CAA cell was 0/10 in both behaviours.**", ""]
    L += ["## Pilot (CAA only, init states 40-49, 10 episodes per cell)", "", "| behaviour | chosen L | grid | best pilot rate |",
          "|---|---|---|---|"]
    for k, p in v["pilot"]["choice"].items():
        L.append(f"| {k} | {p['L']} | {p['grid']} | {p['pilot_best']:.2f} |")
    L += ["", "pilot rates: " + "; ".join(f"{k}: " + ", ".join(f"{c}={r:.1f}" for c, r in p["rates"].items()) for k, p in v["pilot"]["choice"].items()), ""]
    for k, r in v.get("kinds", {}).items():
        L += [f"## {k} ({'task switch: success of task B under instruction A' if k == 'TS' else 'start suppression: max EEF displacement < 2 cm over 100 steps'})",
              "", f"label: **{r['label']}** | n per cell {r['n_per_cell']} | paired items {r['n_paired']} | none {fmt(r['none'])}"
              + (f" | oracle (instruction B) {fmt(r['oracle'])}" if "oracle" in r else "") + f" | validity {'OK' if r['valid'] else 'FAILED'}", "",
              "| arm | " + " | ".join(f"alpha={a}" for a in r["grid"]) + " | significant reversals |", "|---|" + "---|" * 5]
        for arm in ARMS:
            L.append(f"| {arm} | " + " | ".join(fmt(x) for x in r["rates"][arm]) + f" | {r['reversals'][arm]} |")
        L += ["", f"best CAA rate {r['best_caa']:.3f}; paired gain sharp_removed - caa {fmt(r['d_sharp'])}; rand_removed - caa "
              f"{fmt(r['d_rand'])}; sharp_removed - rand_removed {fmt(r['d_sharp_vs_rand'])}", f"H1 criteria: {r['H1']}", ""]
        if "late_start" in r:
            L += ["late start (still <2 cm at step 50, >=2 cm by step 100): " + "; ".join(f"{a}: {np.round(x, 3).tolist()}" for a, x in r["late_start"].items()), ""]
    for u, h in v.get("hessian", {}).items():
        L += [f"### Steering Hessian {u}: L={h['L']}, |v_CAA|={h['caa_norm']:.3f}, sharp energy eps={h['eps_sharp_energy']:.4f}, "
              f"top eigvals {np.round(h['eigvals'][:5], 5).tolist()}", "",
              "open-loop anti-steer (held-out states, fraction with loss up): " + ", ".join(f"{k}={d['anti_steer']:.2f}" for k, d in h["open_loop"].items()), ""]
    readme = open(os.path.join(HERE, "README.md")).read()
    L += ["## Gate (verbatim from README)", "", readme[readme.index("## PRE-REGISTERED GATE"):readme.index("## Files")].strip(), ""]
    open(P("RESULTS.md"), "w").write("\n".join(L))
    if not smoke:
        shutil.copy(P("RESULTS.md"), os.path.join(HERE, "RESULTS.md"))
    print("\n".join(L[:60]))


def analyze(pytest_ok, smoke):
    import vla
    pilot = jload(P("pilot.json"))
    v = dict(date=time.strftime("%Y-%m-%d %H:%M"), smoke=smoke, pytest_ok=pytest_ok, policy=vla.REPO, h_exec=vla.H_EXEC, pilot=pilot)
    if pilot.get("early_stop"):
        v.update(early_stop=True, overall="FAIL" if pytest_ok else "INVALID")
    else:
        v["kinds"] = {k: analyze_kind(k, us, pilot["choice"][k]["grid"]) for k, us in KINDS.items()}
        v["hessian"] = {u: jload(P(f"hess_{u}.json")) for u in vla.UNITS}
        v["overall"] = overall(v["kinds"], pytest_ok)
    jdump(v, P("verdict.json"))
    write_results(v, smoke)
    return v


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="2 episodes, 1 layer/alpha pilot, 20 steps: plumbing only")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if a.analyze_only:
        pytest_ok = jload(P("pytest.json"))["ok"]
    else:
        env = dict(os.environ, SSV_REQUIRE_GPU="1")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_ssv.py"], cwd=HERE, env=env)
        pytest_ok = r.returncode == 0
        jdump(dict(ok=pytest_ok, date=time.strftime("%Y-%m-%d %H:%M")), P("pytest.json"))
        if not pytest_ok:
            raise SystemExit("pytest pre-flight failed on GPU -> campaign aborted (INVALID)")
        import vla
        R = Runner(a.smoke)
        R.pilot()
        ch = {k: pilot_choice(k, read_cells(P(f"pilot_{KINDS[k][0]}.jsonl"))) for k in KINDS}
        pilot = dict(choice=ch, early_stop=(not a.smoke) and all(ch[k]["all_zero"] for k in KINDS))
        jdump(pilot, P("pilot.json"))
        if not pilot["early_stop"]:
            for unit, u in vla.UNITS.items():
                h = R.hessian(unit, ch[u["kind"]]["L"], ch[u["kind"]]["grid"])
                R.main(unit, h)
    analyze(pytest_ok, a.smoke)
