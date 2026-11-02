"""Full pre-registered campaign, resumable: pytest (abort on fail) -> Qwen3.5 run.py -> logit.py -> Qwen2.5 run.py ->
logit.py -> gate analysis (RESULTS/verdict.json + RESULTS/RESULTS.md, copied to repo RESULTS.md).
Usage: python run_all.py [--smoke] [--analyze-only]. Results dir: $RP_RESULTS (default ./results)."""
import sys, os, json, subprocess, re, time, shutil, argparse
import math

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RP_RESULTS", os.path.join(HERE, "results"))
HYB, CTRL = "Qwen/Qwen3.5-0.8B", "Qwen/Qwen2.5-0.5B-Instruct"

def sh(args, log):
    """Run a stage, tee output to RES/<log>; raise on failure."""
    print(f"\n=== {' '.join(args)}", flush=True)
    with open(os.path.join(RES, log), "a") as f:
        p = subprocess.Popen([sys.executable] + args, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); f.write(line); f.flush()
        if p.wait():
            raise SystemExit(f"stage failed ({p.returncode}): {args}")

# ---------------------------------------------------------------- gate (README, frozen 2026-11-02)
def disjoint(a, b):  # a, b = [p, lo, hi]
    return a[1] > b[2] or b[1] > a[2]

def above(a, b):
    return a[1] > b[2]

def q1(s):
    nl, r = s["NL"]["flip"], s["NL_rand"]["flip"]
    if nl[0] >= 0.30 and r[0] <= 0.05 and disjoint(nl, r):
        return "YES"
    return "NO" if nl[0] < 0.10 or not disjoint(nl, r) else "WEAK"

def locus(est, ctrl_of):
    """est: {cond: [p, lo, hi]}. A condition not beating its _rand control counts as 0 (CI [0,0,0])."""
    e = {c: est[c] if above(est[c], est[ctrl_of[c]]) else [0.0, 0.0, 0.0] for c in ("STATE", "KV")}
    st, kv = e["STATE"], e["KV"]
    if st[0] >= 2 * kv[0] and disjoint(st, kv):
        L = "STATE"
    elif kv[0] >= 2 * st[0] and disjoint(st, kv):
        L = "KV"
    else:
        L = "SHARED"
    return L, {c: dict(estimate=est[c], control=est[ctrl_of[c]], beats_control=above(est[c], est[ctrl_of[c]])) for c in e}

def sec_planning(f):
    nl, r = f["NL"], f["NL_rand"]
    if any(math.isnan(x) for x in nl[:3] + r[:3]):
        return "NO DATA"
    if nl[0] >= 0.30 and disjoint(nl[:3], r[:3]):
        return "YES"
    return "NO" if nl[0] < 0.10 else "WEAK"

def analyze(pytest_ok, smoke):
    P = {m: json.load(open(os.path.join(RES, m.split("/")[-1] + ".json"))) for m in (HYB, CTRL)}
    S = {m: json.load(open(os.path.join(RES, m.split("/")[-1] + "_logit.json"))) for m in (HYB, CTRL)}
    ph = P[HYB]["summary"]
    both = ph["BOTH"]["flip"]
    valid = dict(pytest_noop_and_full_patch=pytest_ok, hybrid_BOTH_flip=both[0], hybrid_BOTH_ge_90=both[0] >= 0.90,
                 n_pairs={m: P[m]["n_pairs"] for m in P})
    is_valid = pytest_ok and both[0] >= 0.90 and P[HYB]["n_pairs"] > 0
    prim = {m: dict(Q1=q1(P[m]["summary"]) if P[m]["n_pairs"] else "NO DATA", n_pairs=P[m]["n_pairs"],
                    n_ge_200=P[m]["n_pairs"] >= 200, clean_rhyme=P[m]["clean_rhyme"],
                    NL=P[m]["summary"]["NL"]["flip"] if P[m]["n_pairs"] else None,
                    NL_rand=P[m]["summary"]["NL_rand"]["flip"] if P[m]["n_pairs"] else None) for m in P}
    q2, q2d = locus({c: ph[c]["flip"] for c in ph}, {"STATE": "STATE_rand", "KV": "KV_rand"})
    prim[HYB].update(Q2=q2, Q2_detail=q2d)
    q1h = prim[HYB]["Q1"]
    overall = "INVALID" if not is_valid else "PASS" if q1h == "YES" and q2 in ("STATE", "KV") else "FAIL" if q1h == "NO" else "MIXED"
    fh = S[HYB]["summary"]
    sec = {m: dict(planning=sec_planning(S[m]["summary"]), NL=S[m]["summary"]["NL"], NL_rand=S[m]["summary"]["NL_rand"],
                   n_pairs_kept=S[m]["n_pairs_kept"], BOTH_f=S[m]["summary"].get("BOTH")) for m in S}
    if not any(math.isnan(x) for c in ("STATE", "STATE_rand", "KV", "KV_rand") for x in fh[c][:3]):
        sec[HYB]["locus"], sec[HYB]["locus_detail"] = locus({c: v[:3] for c, v in fh.items()}, {"STATE": "STATE_rand", "KV": "KV_rand"})
    else:
        sec[HYB]["locus"] = "NO DATA"
    if not is_valid:
        for m in sec:
            sec[m]["verdict_label"] = "INVALID"
    v = dict(date=time.strftime("%Y-%m-%d %H:%M"), smoke=smoke, overall=overall, valid=is_valid, validity=valid,
             primary=prim, secondary=sec, q2_interpretation=("which memory pathway carries the rhyme-word information into line 2"
                                                            if q1h != "YES" else "where the plan lives"),
             kernels=P[HYB].get("kernels"), device=P[HYB].get("device"), dtype=P[HYB].get("dtype"))
    json.dump(v, open(os.path.join(RES, "verdict.json"), "w"), indent=1)

    readme = open(os.path.join(HERE, "README.md")).read()
    gate = readme[readme.index("## PRE-REGISTERED GATE"):readme.index("## Files")].strip()
    secg = readme[readme.index("- Secondary gate:"):].split("\n- This secondary")[0].strip()
    fmt = lambda x: "-" if x is None else f"{x[0]:.3f} [{x[1]:.3f}, {x[2]:.3f}]"
    L = [f"# RESULTS - rhyme-plan-hybrid{' (SMOKE TEST, NOT A RESULT)' if smoke else ''}", "",
         f"Generated {v['date']} by run_all.py on {v['device']} ({v['dtype']}), GDN kernels: {v['kernels']}.", "",
         f"## Overall verdict (primary gate): **{overall}**", "",
         f"Validity: pytest no-op/full-patch {'PASS' if pytest_ok else 'FAIL'}; hybrid BOTH flip {fmt(both)} "
         f"({'>=' if both[0] >= 0.9 else '<'} 90%). " + ("" if is_valid else "**Run is INVALID, not a result.**"), "",
         "## Primary: free-generation flip rates (95% Wilson CI)", "",
         "| condition | " + " | ".join(m.split("/")[-1] for m in P) + " |", "|---|" + "---|" * len(P)]
    for c in ph:
        L.append(f"| {c} | " + " | ".join(fmt(P[m]["summary"][c]["flip"]) if c in P[m]["summary"] else "n/a" for m in P) + " |")
    L.append(f"| flip_nocopy NL | " + " | ".join(fmt(P[m]["summary"]["NL"]["flip_nocopy"]) for m in P) + " |")
    L += ["", "| model | n pairs | N>=200 | clean rhyme rate | Q1 |", "|---|---|---|---|---|"]
    for m, d in prim.items():
        L.append(f"| {m} | {d['n_pairs']} | {d['n_ge_200']} | {d['clean_rhyme']:.3f} | **{d['Q1']}** |")
    L += ["", f"Q2 locus (Qwen3.5): **{q2}** - " + "; ".join(f"{c} {fmt(d['estimate'])} vs control {fmt(d['control'])} "
          f"(beats control: {d['beats_control']})" for c, d in q2d.items()),
          f"Q2 interpretation (fixed in advance): {v['q2_interpretation']}.", "",
          "## Secondary: teacher-forced fraction restored f (mean, 95% bootstrap CI)", "",
          "| condition | " + " | ".join(m.split("/")[-1] for m in S) + " |", "|---|" + "---|" * len(S)]
    for c in fh:
        L.append(f"| {c} | " + " | ".join(fmt(S[m]["summary"][c][:3]) if c in S[m]["summary"] else "n/a" for m in S) + " |")
    L += ["", "| model | pairs kept | newline planning | locus |", "|---|---|---|---|"]
    for m, d in sec.items():
        L.append(f"| {m} | {d['n_pairs_kept']} | **{d['planning']}** | {d.get('locus', 'n/a')} |")
    L += ["", "Secondary result can not turn a primary FAIL into a PASS." + ("" if is_valid else " INVALID (validity failed)."), "",
          "Note: a condition that does not beat its `_rand` control (disjoint CIs) counts as 0 with CI [0, 0] in the locus rule.", "",
          "## Gate (verbatim from README)", "", gate, "", secg, ""]
    open(os.path.join(RES, "RESULTS.md"), "w").write("\n".join(L))
    if not smoke:
        shutil.copy(os.path.join(RES, "RESULTS.md"), os.path.join(HERE, "RESULTS.md"))
    print("\n".join(L[:30]))
    return v

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 pairs per stage, any pairs; validates plumbing only")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    pytest_ok = True
    if not a.analyze_only:
        # pre-flight on the run device (fp32): abort the campaign if patching is broken on this kernel path
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_rp.py"], cwd=HERE)
        pytest_ok = r.returncode == 0
        json.dump(dict(ok=pytest_ok, date=time.strftime("%Y-%m-%d %H:%M")), open(os.path.join(RES, "pytest.json"), "w"))
        if not pytest_ok:
            raise SystemExit("pytest failed: patching invalid on this device/kernel path -> campaign aborted (INVALID)")
        ra = ["--nlines", "24", "--npairs", "3", "--any-pairs"] if a.smoke else []
        la = ["--npairs", "3"] if a.smoke else []
        for m in (HYB, CTRL):
            stem = m.split("/")[-1]
            if not os.path.exists(os.path.join(RES, stem + ".json")):
                sh(["run.py", m] + ra, f"run_{stem}.log")
            if not os.path.exists(os.path.join(RES, stem + "_logit.json")):
                sh(["logit.py", m] + la, f"logit_{stem}.log")
    else:
        pytest_ok = json.load(open(os.path.join(RES, "pytest.json")))["ok"]
    analyze(pytest_ok, a.smoke)
