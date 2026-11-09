"""Full pre-registered campaign, resumable (finished rollout units are skipped):
pytest pre-flight -> pilot (base, 40 tasks) -> task selection -> memory write -> dev sweep (+ kill rule) -> test
(base/mem, unrelated tasks, random/neg controls) -> LoRA baseline -> secondary conditions -> gate analysis
(verdict.json + RESULTS.md). Usage: python run_all.py [--analyze-only] [--smoke]. Results dir: $CM_RESULTS."""
import argparse, json, os, shutil, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("CM_RESULTS", os.path.join(HERE, "results"))
MAX_H = float(os.environ.get("CM_MAX_HOURS", "3.6"))      # compute stop (L4 ~3.8 CU/h -> ~13.7 CU), installs excluded
B = 10                                                     # envs per batched rollout
PILOT, DEV, TEST = range(0, 10), range(0, 20), range(20, 50)
ALPHAS, LORA_LRS, SEEDS = (0.5, 1.0, 2.0, 4.0), (3e-5, 1e-4, 3e-4), (0, 1)
N_F, U_PER_F = 4, 2
P = lambda *a: os.path.join(RES, *a)


# ================================================================ gate (README, frozen 2026-11-08)
def gate(prim, sel, rnd, mem_vs_rnd, dev_best, base_F):
    """prim/sel/rnd/mem_vs_rnd = paired [mean diff, lo, hi, n]; dev_best = best dev gain; base_F = base success on F test."""
    if dev_best is not None and dev_best < 0.03:
        return "FAIL", "killed at dev: gain < 3 pts at every (site, alpha)"
    if base_F is not None and base_F > 0.85:
        return "INVALID", "base success on the corrected family > 85% on test: no headroom"
    if prim is None or np.isnan(prim[0]):
        return "INCOMPLETE", "primary units missing"
    if prim[0] < 0.03:
        return "FAIL", "test gain < 3 pts"
    ok = (prim[0] >= 0.10 and prim[1] > 0 and sel is not None and sel[0] >= -0.02
          and rnd is not None and rnd[0] <= 0.03 and mem_vs_rnd is not None and mem_vs_rnd[1] > 0)
    return ("PASS", "all primary criteria met") if ok else ("MIXED", "gain >= 3 pts but not every PASS criterion")


# ================================================================ bookkeeping
def load_units():
    out = {}
    if os.path.exists(P("episodes.jsonl")):
        for line in open(P("episodes.jsonl")):
            r = json.loads(line)
            out[r["key"]] = r
    return out


def jload(name, default=None):
    return json.load(open(P(name))) if os.path.exists(P(name)) else default


def jsave(name, obj):
    json.dump(obj, open(P(name), "w"), indent=1)


class Budget(Exception):
    pass


class Clock:
    def __init__(self):
        self.spent = jload("clock.json", {"s": 0.0})["s"]

    def add(self, s):
        self.spent += s
        jsave("clock.json", {"s": self.spent})

    def check(self):
        if self.spent > MAX_H * 3600:
            raise Budget(f"compute budget {MAX_H} h reached")


# ================================================================ campaign
class Campaign:
    def __init__(self, smoke=False):
        import torch, cm
        self.torch, self.cm, self.smoke = torch, cm, smoke
        self.units, self.clock = load_units(), Clock()
        self.policy, self.pre, self.post = cm.load_policy()
        self.mem = cm.Memory(self.policy)
        self.sites = self.mem.sites
        self.env, self.env_key, self.ds = None, None, None
        self.lora_cache = {}

    # ---- environments
    def get_env(self, task):
        if self.env_key != task:
            from lerobot.envs.factory import make_env_pre_post_processors
            if self.env is not None:
                self.env.close()
            suite, tid = task.split(":")
            self.env, cfg = self.cm.make_task_env(suite, int(tid), B)
            self.envproc = make_env_pre_post_processors(cfg, self.policy.config)
            self.env_key = task
        return self.env

    def task_desc(self, task):
        from libero.libero import benchmark
        suite, tid = task.split(":")
        return benchmark.get_benchmark_dict()[suite]().get_task(int(tid)).language

    # ---- one rollout unit (B episodes); resumable
    def unit(self, cond, task, seed, ids, ctx, rephrase=False):
        ids = list(ids)
        out = []
        for c in range(0, len(ids), B):
            chunk = ids[c:c + B]
            if self.smoke:
                chunk = chunk[:2] + [chunk[0]] * (B - 2)
            key = f"{cond}|{task}|s{seed}|{chunk[0]}"
            if key not in self.units:
                self.clock.check()
                t0 = time.time()
                env = self.get_env(task)
                pre = self.cm.Rephrased(self.pre) if rephrase else self.pre
                with ctx():
                    succ = self.cm.run_episodes(env, self.policy, (*self.envproc, pre, self.post), chunk, 1000 * seed)
                    fired = None if self.mem.fired is None else self.mem.fired.tolist()
                dt = time.time() - t0
                r = dict(key=key, cond=cond, task=task, seed=seed, ids=chunk, success=succ, fired=fired, sec=dt)
                with open(P("episodes.jsonl"), "a") as f:
                    f.write(json.dumps(r) + "\n")
                self.units[key] = r
                self.clock.add(dt)
                print(f"[unit] {key}: {sum(succ)}/{len(succ)} ({dt:.0f}s, total {self.clock.spent / 3600:.2f} h)", flush=True)
            out.append(self.units[key])
        return out

    def off(self):
        return self.mem.using("off")

    # ---- stage 1: pilot + selection
    def pilot(self):
        from libero.libero import benchmark
        tasks = [f"{s}:{t}" for s in self.cm.SUITES for t in range(len(benchmark.get_benchmark_dict()[s]().tasks))]
        if self.smoke:
            tasks = tasks[::10]
        for t in tasks:
            self.unit("base", t, 0, PILOT, self.off)
        old = jload("selection.json")
        # reuse, unless it was cut to "lite" under a smaller compute cap than the current one (Colab 3.6 h -> 24 h)
        if old and not (old["lite"] and old["eta_h"] <= MAX_H):
            return old
        rate = {t: np.mean(sum((self.units[f"base|{t}|s0|{i}"]["success"] for i in PILOT[::B]), [])) for t in tasks}
        order = sorted(tasks, key=lambda t: (rate[t], tasks.index(t)))
        mid = [t for t in order if 0.05 < rate[t] <= 0.75]
        F = (mid + [t for t in order if rate[t] <= 0.05 and t not in mid])[:N_F]
        U = []
        for f in F:
            s, i = f.split(":")
            same = sorted((t for t in tasks if t.startswith(s + ":") and t not in F and t not in U),
                          key=lambda t: (abs(int(t.split(":")[1]) - int(i)), int(t.split(":")[1])))
            U += same[:U_PER_F]
        sec_ep = np.mean([self.units[f"base|{t}|s0|0"]["sec"] for t in tasks]) / B
        n_left = (len(F) * len(DEV) * (1 + 2 * len(ALPHAS) + len(LORA_LRS)) + len(F) * len(TEST) * (2 * 2 + 2 + 2 + 3)
                  + len(U) * len(TEST) * 2)
        eta_h = sec_ep * n_left / 3600 + self.clock.spent / 3600
        lite = eta_h > MAX_H
        sel = dict(F=F, U=U[: (len(F) if lite else len(U))], pilot_rate=rate, sec_per_episode=sec_ep, eta_h=eta_h, lite=lite,
                   desc={t: self.task_desc(t) for t in F + U})
        jsave("selection.json", sel)
        print("[select]", json.dumps({k: sel[k] for k in ("F", "U", "eta_h", "lite")}), flush=True)
        return sel

    # ---- stage 2: memories (no gradient step on weights)
    def build_memories(self, sel):
        torch, cm = self.torch, self.cm
        if os.path.exists(P("memory.pt")):
            return torch.load(P("memory.pt"), weights_only=False)
        t0 = time.time()
        self.ds = self.ds or cm.load_dataset()
        ept = episodes_by_task_norm(cm.episodes_by_task(self.ds))
        # whitening pool: first frames of 2 demos for every dataset task
        keys, labels = [], []
        for li, (task, eps) in enumerate(sorted(ept.items())):
            for (_, a, _) in eps[:2]:
                with self.mem.using("off"), torch.no_grad():
                    b = cm.frames_batch(self.ds, [a], self.pre)
                    cm.own_chunk(self.policy, b, torch.zeros_like(self.policy.prepare_action(b)))
                keys.append(self.mem.last_key.cpu())
                labels.append(li)
        keys = torch.cat(keys)
        W = cm.Whitener(keys)
        tau = cm.pick_tau(W(keys), labels)
        out = dict(whiten=W, tau=tau, mem={}, demos={})
        for s in SEEDS:
            for t in sel["F"]:
                eps = ept[norm(sel["desc"][t])]
                rng = np.random.default_rng(1000 * s + sel["F"].index(t))
                ep, a, z = eps[int(rng.integers(len(eps)))]
                idx = list(range(a, z, 4))[:32]
                b = cm.frames_batch(self.ds, idx, self.pre)
                sd, so, _ = cm.demo_shift(self.policy, self.mem, b, seed=s)
                with self.mem.using("off"), torch.no_grad():
                    b0 = cm.frames_batch(self.ds, [a], self.pre)
                    cm.own_chunk(self.policy, b0, torch.zeros_like(self.policy.prepare_action(b0)))
                key = self.mem.last_key.cpu()
                g = torch.Generator().manual_seed(7 + 1000 * s + sel["F"].index(t))
                d = {}
                for site in self.sites:
                    own = (sd[site] - so[site]).mean(0).cpu()
                    V, ev = cm.gn_sharp_dirs(self.policy, self.mem, cm.frames_batch(self.ds, idx[:4], self.pre), site, seed=s)
                    rnd = torch.randn(own.shape, generator=g)
                    d[site] = dict(own=own, neg=-own, rand=rnd / rnd.norm() * own.norm(), sharp=cm.remove_dirs(own, V.cpu()),
                                   demo_mean=sd[site].mean(0).cpu(), sharp_frac=float((V.T @ own).norm() / own.norm()),
                                   gn_top=ev[:8].tolist())
                out["mem"][(s, t)] = dict(key=key, d=d)
                out["demos"][(s, t)] = dict(episode=ep, frames=[a, z], n_used=len(idx))
        for s in SEEDS:   # reference = mean teacher-forced state of the OTHER corrected tasks' demos
            for t in sel["F"]:
                for site in self.sites:
                    others = [out["mem"][(s, u)]["d"][site]["demo_mean"] for u in sel["F"] if u != t]
                    out["mem"][(s, t)]["d"][site]["refmean"] = out["mem"][(s, t)]["d"][site]["demo_mean"] - torch.stack(others).mean(0)
        torch.save(out, P("memory.pt"))
        z = W(torch.stack([out["mem"][(0, t)]["key"][0] for t in sel["F"]]))
        jsave("memory_summary.json", dict(tau=tau, demos={f"{s}|{t}": v for (s, t), v in out["demos"].items()},
                                          norms={f"{s}|{t}|{site}": float(v["d"][site]["own"].norm()) for (s, t), v in out["mem"].items() for site in self.sites},
                                          sharp_frac={f"{s}|{t}|{site}": v["d"][site]["sharp_frac"] for (s, t), v in out["mem"].items() for site in self.sites},
                                          F_key_cos=(z @ z.T).tolist(), sec=time.time() - t0))
        self.clock.add(time.time() - t0)
        return out

    def bank(self, M, seed, site, variant, F):
        W = M["whiten"]
        return [(W(M["mem"][(seed, t)]["key"])[0], M["mem"][(seed, t)]["d"][site][variant]) for t in F]

    def mem_ctx(self, M, seed, site, alpha, variant, F):
        bank = self.bank(M, seed, site, variant, F)
        def ctx():
            self.mem.whiten, self.mem.tau = M["whiten"], M["tau"]
            return self.mem.using("gated", bank=bank, alpha=alpha, site=site)
        return ctx

    # ---- LoRA baseline (one demo, tuned lr on dev)
    def lora_ctx(self, sel, task, seed, lr):
        torch, cm = self.torch, self.cm
        f = P(f"lora_{task.replace(':', '_')}_s{seed}_lr{lr:g}.pt")
        if not os.path.exists(f):
            self.clock.check()
            t0 = time.time()
            self.ds = self.ds or cm.load_dataset()
            M = torch.load(P("memory.pt"), weights_only=False)
            a, z = M["demos"][(seed, task)]["frames"]
            b = cm.frames_batch(self.ds, list(range(a, z)), self.pre)
            deltas, losses = cm.lora_finetune(self.policy, b, lr, seed=seed)
            torch.save(dict(deltas={k: v.cpu() for k, v in deltas.items()}, losses=losses), f)
            self.clock.add(time.time() - t0)
        deltas = torch.load(f)["deltas"]
        dev = next(self.policy.parameters()).device
        return lambda: cm.lora_applied(self.policy, {k: v.to(dev) for k, v in deltas.items()})

    def run(self):
        sel = self.pilot()
        F, U = sel["F"], sel["U"]
        M = self.build_memories(sel)
        # dev sweep (correction seed 0)
        for t in F:
            self.unit("base", t, 0, DEV, self.off)
            for site in self.sites:
                for al in ALPHAS:
                    self.unit(f"mem:{site}:{al:g}:own", t, 0, DEV, self.mem_ctx(M, 0, site, al, "own", F))
        dev = dev_choice(self.units, F, self.sites)
        jsave("dev.json", dev)
        if dev["best_gain"] < 0.03:
            print("[kill] dev gain < 3 pts at every (site, alpha): campaign stops (FAIL)", flush=True)
            return
        site, al = dev["site"], dev["alpha"]
        mem_name = lambda v: f"mem:{site}:{al:g}:{v}"
        for s in SEEDS:                                   # primary
            for t in F:
                self.unit("base", t, s, TEST, self.off)
                self.unit(mem_name("own"), t, s, TEST, self.mem_ctx(M, s, site, al, "own", F))
        for t in U:                                       # selectivity
            self.unit("base", t, 0, TEST, self.off)
            self.unit(mem_name("own"), t, 0, TEST, self.mem_ctx(M, 0, site, al, "own", F))
        for v in ("rand", "neg"):                         # controls
            for t in F:
                self.unit(mem_name(v), t, 0, TEST, self.mem_ctx(M, 0, site, al, v, F))
        for t in F:                                       # LoRA: lr tuned on dev, then test
            for lr in LORA_LRS:
                self.unit(f"lora:{lr:g}", t, 0, DEV, self.lora_ctx(sel, t, 0, lr))
        lr = lora_choice(self.units, F)
        jsave("lora_dev.json", dict(lr=lr))
        for s in SEEDS:
            for t in F:
                self.unit(f"lora:{lr:g}", t, s, TEST, self.lora_ctx(sel, t, s, lr))
        ids2 = TEST[:10] if sel["lite"] else TEST         # secondary conditions (seed 0)
        for t in F:
            self.unit(mem_name("refmean"), t, 0, ids2, self.mem_ctx(M, 0, site, al, "refmean", F))
            self.unit(mem_name("sharp"), t, 0, ids2, self.mem_ctx(M, 0, site, al, "sharp", F))
            self.unit("rephrase", t, 0, ids2, self.off, rephrase=True)
        if self.env is not None:
            self.env.close()


def norm(s):
    return " ".join(s.lower().replace("_", " ").split())


def episodes_by_task_norm(d):
    return {norm(k): v for k, v in d.items()}


# ================================================================ analysis
def eps(units, cond, tasks, seeds, ids):
    """{(task, seed, init_id): success} for the requested cells."""
    out = {}
    for r in units.values():
        if r["cond"] == cond and r["task"] in tasks and r["seed"] in seeds:
            for i, s in zip(r["ids"], r["success"]):
                if i in ids:
                    out[(r["task"], r["seed"], i)] = s
    return out


def pair(units, a, b, tasks, seeds, ids, b_seed0=False):
    A = eps(units, a, tasks, seeds, ids)
    Bd = eps(units, b, tasks, (0,) if b_seed0 else seeds, ids)
    ks = [k for k in A if (k if not b_seed0 else (k[0], 0, k[2])) in Bd]
    import cm
    return cm.paired_boot([A[k] for k in ks], [Bd[k if not b_seed0 else (k[0], 0, k[2])] for k in ks]) if ks else None


def dev_choice(units, F, sites):
    gains = {}
    for site in sites:
        for al in ALPHAS:
            p = pair(units, f"mem:{site}:{al:g}:own", "base", F, (0,), DEV)
            gains[f"{site}:{al:g}"] = p
    best = max(gains, key=lambda k: (gains[k][0], -float(k.split(":")[1]), k.startswith("out")))
    return dict(gains=gains, site=best.split(":")[0], alpha=float(best.split(":")[1]), best_gain=gains[best][0])


def lora_choice(units, F):
    g = {lr: pair(units, f"lora:{lr:g}", "base", F, (0,), DEV)[0] for lr in LORA_LRS}
    return max(LORA_LRS, key=lambda lr: (g[lr], -lr))


def rate(units, cond, tasks, seeds, ids):
    import cm
    v = list(eps(units, cond, tasks, seeds, ids).values())
    return cm.wilson(sum(v), len(v)) + [len(v)]


def analyze(smoke=False):
    units, sel, dev = load_units(), jload("selection.json"), jload("dev.json")
    pre = jload("pytest.json", {"ok": False})
    if sel is None:
        raise SystemExit("no selection.json: pilot not run")
    F, U = sel["F"], sel["U"]
    fmt = lambda x: "-" if x is None or (isinstance(x, list) and np.isnan(x[0])) else f"{100 * x[0]:+.1f} [{100 * x[1]:+.1f}, {100 * x[2]:+.1f}] (n={x[3]})"
    fr = lambda x: f"{100 * x[0]:.1f}% [{100 * x[1]:.1f}, {100 * x[2]:.1f}] (n={x[3]})" if x[3] else "-"
    res = dict(F=F, U=U, lite=sel["lite"], pytest_ok=pre["ok"], dev=dev)
    if dev is None or dev["best_gain"] < 0.03 or not pre["ok"]:
        prim = sel_ = rnd = mvr = base_F = None
        lr = None
    else:
        m = lambda v: f"mem:{dev['site']}:{dev['alpha']:g}:{v}"
        prim = pair(units, m("own"), "base", F, SEEDS, TEST)
        sel_ = pair(units, m("own"), "base", U, (0,), TEST)
        rnd = pair(units, m("rand"), "base", F, (0,), TEST)
        mvr = pair(units, m("own"), m("rand"), F, (0,), TEST)
        base_F = rate(units, "base", F, SEEDS, TEST)[0] if rate(units, "base", F, SEEDS, TEST)[3] else None
        lr = (jload("lora_dev.json") or {}).get("lr")
        lora = pair(units, f"lora:{lr:g}", "base", F, SEEDS, TEST) if lr is not None else None
        res.update(primary=prim, selectivity=sel_, rand=rnd, mem_minus_rand=mvr, base_F=base_F,
                   neg=pair(units, m("neg"), "base", F, (0,), TEST), lora_lr=lr, lora=lora,
                   mem_minus_lora=pair(units, m("own"), f"lora:{lr:g}", F, SEEDS, TEST) if lr is not None else None,
                   refmean=pair(units, m("refmean"), "base", F, (0,), TEST),
                   sharp=pair(units, m("sharp"), "base", F, (0,), TEST),
                   sharp_minus_own=pair(units, m("sharp"), m("own"), F, (0,), TEST),
                   refmean_minus_own=pair(units, m("refmean"), m("own"), F, (0,), TEST),
                   rephrase=pair(units, "rephrase", "base", F, (0,), TEST),
                   rates={c: rate(units, c, F, SEEDS, TEST) for c in ["base", m("own"), f"lora:{lr:g}" if lr else "none"]},
                   rates_U={c: rate(units, c, U, (0,), TEST) for c in ["base", m("own")]})
        # gate firing: which bank entry fired per episode (bank order = F)
        fire = {"F_correct": [], "U_fired": []}
        for r in units.values():
            if r["cond"] == m("own") and r["fired"] is not None and r["ids"][0] in TEST:
                if r["task"] in F:
                    fire["F_correct"] += [f == F.index(r["task"]) for f in r["fired"]]
                elif r["task"] in U:
                    fire["U_fired"] += [f >= 0 for f in r["fired"]]
        res["gate_recall_F"] = float(np.mean(fire["F_correct"])) if fire["F_correct"] else None
        res["gate_false_fire_U"] = float(np.mean(fire["U_fired"])) if fire["U_fired"] else None
    overall, why = gate(prim, sel_, rnd, mvr, dev["best_gain"] if dev else None, base_F)
    if not pre["ok"]:
        overall, why = "INVALID", "pytest pre-flight failed"
    res.update(overall=overall, why=why, date=time.strftime("%Y-%m-%d %H:%M"), smoke=smoke,
               compute_h=jload("clock.json", {"s": 0})["s"] / 3600)
    jsave("verdict.json", res)

    readme = open(os.path.join(HERE, "README.md")).read()
    gtxt = readme[readme.index("## PRE-REGISTERED GATE"):readme.index("## Files")].strip()
    L = [f"# RESULTS - correction-memory{' (SMOKE TEST, NOT A RESULT)' if smoke else ''}", "",
         f"Generated {res['date']} by run_all.py. Compute used: {res['compute_h']:.2f} h. Lite mode: {sel['lite']}.", "",
         f"## Overall verdict: **{overall}** ({why})", "",
         f"Corrected family F: {F}", f"Unrelated tasks U: {U}", "",
         "Pilot base success (init states 0-9): " + ", ".join(f"{t} {100 * v:.0f}%" for t, v in sel["pilot_rate"].items()), ""]
    if dev:
        L += ["## Dev sweep (F, init states 0-19, correction seed 0): gain vs base, pts [95% paired bootstrap CI]", "",
              "| site:alpha | gain |", "|---|---|"] + [f"| {k} | {fmt(v)} |" for k, v in dev["gains"].items()]
        L += ["", f"Chosen: site {dev['site']}, alpha {dev['alpha']:g} (best dev gain {100 * dev['best_gain']:+.1f} pts)", ""]
    if "primary" in res:
        L += ["## Test (init states 20-49): paired differences in success, pts [95% bootstrap CI]", "",
              "| comparison | value |", "|---|---|"]
        for k in ("primary", "selectivity", "rand", "mem_minus_rand", "neg", "lora", "mem_minus_lora", "refmean",
                  "refmean_minus_own", "sharp", "sharp_minus_own", "rephrase"):
            L.append(f"| {k} | {fmt(res[k])} |")
        L += ["", "| success rate (Wilson 95%) | value |", "|---|---|"]
        L += [f"| F {c} | {fr(v)} |" for c, v in res["rates"].items()] + [f"| U {c} | {fr(v)} |" for c, v in res["rates_U"].items()]
        L += ["", f"Base success on F test: {res['base_F']}; LoRA lr (dev-tuned): {res['lora_lr']}; "
              f"gate recall on F (correct memory fired): {res['gate_recall_F']}; false-fire rate on U: {res['gate_false_fire_U']}.", ""]
    L += ["## Gate (verbatim from README)", "", gtxt, ""]
    open(P("RESULTS.md"), "w").write("\n".join(L))
    if not smoke:
        shutil.copy(P("RESULTS.md"), os.path.join(HERE, "RESULTS.md"))
    print("\n".join(L[:40]))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 distinct episodes per unit, 4 pilot tasks; plumbing only")
    a = ap.parse_args()
    if a.smoke:
        RES = os.environ.get("CM_RESULTS", os.path.join(HERE, "results")) + "_smoke"
    os.makedirs(RES, exist_ok=True)
    if not a.analyze_only:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_cm.py"], cwd=HERE,
                           env={**os.environ, "CM_REAL": "1"})
        jsave("pytest.json", dict(ok=r.returncode == 0, date=time.strftime("%Y-%m-%d %H:%M")))
        if r.returncode:
            raise SystemExit("pytest pre-flight failed on this device -> campaign aborted (INVALID)")
        try:
            Campaign(smoke=a.smoke).run()
        except Budget as e:
            print(f"[budget] {e}: stopping, analysing what exists", flush=True)
    analyze(a.smoke)
