"""redispatch-break campaign. Stages (run in order):
    python run_all.py data    download + parse + integrity checks
    python run_all.py dev     development run on delivery days <= 2025-04-27 only; writes dev.json
    python run_all.py test    full run; needs dev.json and the frozen gate in README.md; writes RESULTS.md
The gate is computed by `verdict` from README.md "PRE-REGISTERED GATE"."""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "2")   # 4 workers x 2 threads: leaves the laptop usable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import rb

EVAL_START = pd.Timestamp("2024-01-01")
SHOCK = (pd.Timestamp("2025-05-01"), pd.Timestamp("2025-06-29"))     # first 60 scored days after the break
POST = (pd.Timestamp("2025-06-30"), pd.Timestamp("2026-10-31"))
CF_YEAR = (pd.Timestamp("2025-05-01"), pd.Timestamp("2026-04-30"))
PLACEBO = ("2024-04-27", "2024-05-01", "2025-04-27")                 # train end, test start, test end
GROUPS, MODELS = ["lags", "wx", "mkt"], ["ridge", "gbm", "gbm_d"]
SCHEMES = ["expanding", "roll90", "roll365", "ew30", "ew90", "dummy"]
NAIVE = ["naive_last", "naive_week", "naive_m7"]
OUT = rb.DATA


def predictions(df):
    """All forecasts for [EVAL_START, last day]: one column per group|model|scheme, plus the naive ones."""
    cfg = [(g, m, s) for g in GROUPS for m in MODELS for s in SCHEMES]
    cols = Parallel(n_jobs=4)(delayed(rb.backtest)(df, g, m, s, EVAL_START, df.index.max()) for g, m, s in cfg)
    P = pd.DataFrame({"|".join(c): p for c, p in zip(cfg, cols)})
    return P.join(rb.naive(df).reindex(P.index))


def table(P, y, a, b):
    """MAE per column over scored days in [a, b]; the blackout and restoration days are never scored."""
    idx = P.index[(P.index >= a) & (P.index <= b) & ~((P.index >= rb.BREAK) & (P.index <= rb.BREAK + pd.Timedelta(days=2)))]
    idx = idx[y.reindex(idx).notna()]
    return {c: rb.mae(y[idx], P.loc[idx, c]) for c in P.columns}, idx


def best(maes, prefix):
    cand = {c: v for c, v in maes.items() if c.startswith(prefix) and np.isfinite(v)}
    return min(cand, key=cand.get)


def dev():
    df = rb.load(until=rb.DEV_END)
    assert df.index.max() <= rb.DEV_END
    P = predictions(df)
    maes, idx = table(P, df.y, EVAL_START, rb.DEV_END)
    sel = {"naive": best(maes, "naive"), "t0": best({c: v for c, v in maes.items() if c.split("|")[0] in ("lags", "wx")}, ""),
           "t1": best(maes, "mkt|")}
    y = df.y[idx]
    out = {"n_days": len(idx), "mae": maes, "selected": sel,
           "skill_t0": rb.skill(y, P.loc[idx, sel["t0"]], P.loc[idx, sel["naive"]]),
           "skill_t1": rb.skill(y, P.loc[idx, sel["t1"]], P.loc[idx, sel["naive"]]),
           "placebo": {m: rb.counterfactual(df, *PLACEBO, model=m) for m in ("gbm", "ridge")},
"y_mean": float(y.mean()), "y_std": float(y.std())}
    rb.dump(out, f"{OUT}/dev.json")
    P.to_csv(f"{OUT}/dev_predictions.csv")
    top = sorted(maes.items(), key=lambda kv: kv[1])[:12]
    print(f"DEV {idx[0].date()}..{idx[-1].date()} n={len(idx)} mean y {out['y_mean']:.1f} GWh")
    print("\n".join(f"  {c:28s} MAE {v:.2f}" for c, v in top + [(c, maes[c]) for c in NAIVE]))
    print("selected", sel, f"skill t0 {out['skill_t0']:.3f} t1 {out['skill_t1']:.3f}")
    print("placebo", {m: (round(v["ratio"], 3), [round(x, 3) for x in v["ratio_ci"]]) for m, v in out["placebo"].items()})


def verdict(r):
    """README gate, verbatim logic. `r` holds the numbers computed in `test`."""
    v = {"V1_matches_redata": r["integrity"]["redata_within_2pct_frac"] >= 0.95,
         "V2_positive_control": r["dev_skill_t1"] >= 0.15,
         "V3_placebo": 0.80 <= r["placebo_ratio"] <= 1.25,
         "V4_coverage": r["test_target_coverage"] >= 0.95}
    h = {"H1_break": r["cf_ratio"] >= 1.5 and r["cf_ratio_ci"][0] > 1.2,
         "H2_breakdown": r["breakdown_ratio"] >= 1.5,
         "H2_adaptation": r["recovery_expanding"] - r["recovery_ew30"] >= 14,
         "H3_skill": r["h3_skill"] >= 0.10 and r["h3_dm_p"] < 0.05 and r["h3_dm_t"] < 0}
    n = sum([h["H1_break"], h["H2_breakdown"] and h["H2_adaptation"], h["H3_skill"]])
    overall = "INVALID" if not all(v.values()) else ["FAIL", "MIXED", "MIXED", "PASS"][n]
    return {"validity": v, "hypotheses": h, "overall": overall}


def test():
    import json
    readme = open(os.path.join(rb.HERE, "README.md")).read()
    assert "PRE-REGISTERED GATE (frozen" in readme, "freeze the gate in README.md before the test run"
    d = json.load(open(f"{OUT}/dev.json"))
    sel = d["selected"]
    df = rb.load()
    P = predictions(df)
    y = df.y.reindex(P.index)
    ml_t0 = [c for c in P.columns if c.split("|")[0] in ("lags", "wx")]
    P["online_t0"] = rb.online_select(P, y, ml_t0)
    P["online_t1"] = rb.online_select(P, y, [c for c in P.columns if c.startswith("mkt|")])
    P["online_naive"] = rb.online_select(P, y, NAIVE)
    P.to_csv(f"{OUT}/predictions.csv")

    periods = {"dev": (EVAL_START, rb.DEV_END), "shock": SHOCK, "post": POST}
    T = {k: table(P, y, *ab) for k, ab in periods.items()}
    post_idx = T["post"][1]
    naive_post = best(T["post"][0], "naive_")        # hindsight-best naive on the test window: favours the baseline
    exp, ew = sel["t0"].rsplit("|", 1)[0] + "|expanding", sel["t0"].rsplit("|", 1)[0] + "|ew30"
    dev180 = table(P, y, rb.DEV_END - pd.Timedelta(days=179), rb.DEV_END)[0]
    t0 = rb.BREAK + pd.Timedelta(days=3)
    yp, a, b = y[post_idx], P.loc[post_idx, "online_t0"], P.loc[post_idx, naive_post]
    dm_t, dm_p = rb.dm_test(yp, a, b)
    cf = {m: rb.counterfactual(df, rb.DEV_END, *CF_YEAR, model=m) for m in ("gbm", "ridge")}
    scored = pd.date_range(SHOCK[0], POST[1])
    r = {"integrity": json.load(open(f"{OUT}/integrity.json")), "dev_skill_t1": d["skill_t1"], "placebo_ratio": d["placebo"]["gbm"]["ratio"],
         "test_target_coverage": float(y.reindex(scored).notna().mean()),
         "cf_ratio": cf["gbm"]["ratio"], "cf_ratio_ci": cf["gbm"]["ratio_ci"],
         "breakdown_ratio": T["shock"][0][exp] / dev180[exp],
         "recovery_expanding": rb.recovery_day(y, P[exp], P["naive_last"], t0),
         "recovery_ew30": rb.recovery_day(y, P[ew], P["naive_last"], t0),
         "h3_skill": rb.skill(yp, a, b), "h3_dm_t": dm_t, "h3_dm_p": dm_p,
         "h3_skill_ci": rb.boot_ci(lambda i: rb.skill(yp.iloc[i], a.iloc[i], b.iloc[i]), len(yp)),
         "h3_naive": naive_post, "exp_cfg": exp, "ew_cfg": ew}
    v = verdict(r)
    rec = {c: rb.recovery_day(y, P[c], P["naive_last"], t0) for c in P.columns if c.startswith(sel["t0"].rsplit("|", 1)[0])}
    t1_skill = rb.skill(yp, P.loc[post_idx, "online_t1"], b)
    types = df.filter(like="rdup_type_").fillna(0) / 1000
    techs = df.filter(like="rdup_tech_").fillna(0) / 1000
    pre, post = (types.index >= "2024-05-01") & (types.index <= "2025-04-27"), (types.index >= CF_YEAR[0]) & (types.index <= CF_YEAR[1])
    split = {"type": {c[10:]: [float(types.loc[pre, c].mean()), float(types.loc[post, c].mean())] for c in types},
             "tech": {c[10:]: [float(techs.loc[pre, c].mean()), float(techs.loc[post, c].mean())] for c in techs}}
    rb.dump({"numbers": r, "verdict": v, "mae": {k: t[0] for k, t in T.items()}, "counterfactual": cf,
             "recovery_days": rec, "t1_post_skill": t1_skill, "split_gwh_per_day_pre_post": split,
             "selected": sel, "n": {k: len(t[1]) for k, t in T.items()}}, f"{OUT}/verdict.json")

    L = [f"# RESULTS - redispatch-break", "",
         f"Generated {pd.Timestamp.now():%Y-%m-%d %H:%M} by run_all.py. Data through {df.index.max().date()}.", "",
         f"## Overall verdict: **{v['overall']}**", "",
         "Validity: " + ", ".join(f"{k}={x}" for k, x in v["validity"].items()),
         "Hypotheses: " + ", ".join(f"{k}={x}" for k, x in v["hypotheses"].items()), "",
         "## Registered numbers", "",
         f"- H1 actual / old-regime counterfactual, {CF_YEAR[0].date()}..{CF_YEAR[1].date()}: {r['cf_ratio']:.2f} "
         f"[{r['cf_ratio_ci'][0]:.2f}, {r['cf_ratio_ci'][1]:.2f}] ({cf['gbm']['actual_gwh']:.0f} vs "
         f"{cf['gbm']['counterfactual_gwh']:.0f} GWh); ridge counterfactual ratio {cf['ridge']['ratio']:.2f}. "
         f"Placebo ratio {r['placebo_ratio']:.2f}.",
         f"- H2 breakdown: `{exp}` MAE in the shock window / its MAE in the last 180 dev days = {r['breakdown_ratio']:.2f}.",
         f"- H2 adaptation: days to recover vs `naive_last`: expanding {r['recovery_expanding']}, ew30 {r['recovery_ew30']}.",
         f"- H3 pre-market skill, post window (n={len(yp)}): `online_t0` vs `{naive_post}` = {r['h3_skill']:.3f} "
         f"[{r['h3_skill_ci'][0]:.3f}, {r['h3_skill_ci'][1]:.3f}], DM t {dm_t:.2f}, p {dm_p:.4f}.",
         f"- Not gated: post-market `online_t1` skill vs the same baseline {t1_skill:.3f}.", "",
         "## MAE (GWh/day) by period", "", "| forecast | dev | shock | post |", "|---|---|---|---|"]
    show = NAIVE + ["online_naive", "online_t0", "online_t1"] + sorted(c for c in P.columns if "|" in c)
    L += [f"| {c} | {T['dev'][0][c]:.2f} | {T['shock'][0][c]:.2f} | {T['post'][0][c]:.2f} |" for c in show]
    L += ["", f"Scored days: dev {len(T['dev'][1])}, shock {len(T['shock'][1])}, post {len(post_idx)}. "
          f"Mean target: dev {y[T['dev'][1]].mean():.1f}, shock {y[T['shock'][1]].mean():.1f}, post {yp.mean():.1f} GWh/day.",
          "", "## Recovery days after the break, by scheme (model family selected on dev)", ""]
    L += [f"- `{c}`: {n}" for c, n in rec.items()]
    L += ["", "## Phase-I upward redispatch, GWh/day, 2024-05-01..2025-04-27 vs 2025-05-01..2026-04-30", "",
          "| split | key | before | after |", "|---|---|---|---|"]
    L += [f"| {s} | {k} | {a:.2f} | {b:.2f} |" for s in ("type", "tech") for k, (a, b) in split[s].items() if max(a, b) >= 0.05]
    gate = readme[readme.index("## PRE-REGISTERED GATE (frozen"):]
    L += ["", "## Gate (verbatim from README)", "", gate.split("\n## ", 1)[0].strip()]
    with open(os.path.join(rb.HERE, "RESULTS.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L[:22]))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else ""
    if stage == "data":
        rb.download()
        df = rb.build()
        info = rb.integrity(df, rb.redata_monthly(range(rb.START.year, df.index.max().year + 1)))
        rb.dump(info, f"{OUT}/integrity.json")
        print(info)
    elif stage == "dev":
        dev()
    elif stage == "test":
        test()
    else:
        sys.exit(__doc__)
