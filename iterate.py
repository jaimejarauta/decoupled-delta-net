# Exploratory iteration after the frozen test (2026-11-14). Not gated: the post window was already scored once,
# so anything chosen here must be confirmed on days after 2026-10-31 under a new frozen gate.
import sys, pandas as pd, numpy as np
from joblib import Parallel, delayed
import rb

HOL = pd.to_datetime([f"{y}-{md}" for y in (2023, 2024, 2025, 2026, 2027)
                      for md in ("01-01", "01-06", "05-01", "08-15", "10-12", "11-01", "12-06", "12-08", "12-25")]
                     + ["2023-04-07", "2024-03-29", "2025-04-18", "2026-04-03", "2027-03-26"])  # national + Good Friday
COMP = ["rdup_type_cot", "rdup_type_sot", "rdup_type_sca", "rdup_type_rtd", "rdup_type_rsi",
        "rdup_tech_ciclo_combinado", "rdup_tech_nuclear"]


BASE_FEATURES = rb.features


def features(df, group, extra):
    X = BASE_FEATURES(df, group)
    if "hol" in extra:
        h = df.index.isin(HOL)
        X["hol"], X["hol_l1"] = h.astype(float), pd.Series(h, index=df.index).shift(1).fillna(False).astype(float)
        X["hol_n1"] = pd.Series(h, index=df.index).shift(-1).fillna(False).astype(float)  # calendar is known ahead
    if "comp" in extra:
        for c in COMP:
            v = df[c].fillna(0) / 1e3 if c in df else 0.0
            X[c + "_l1"], X[c + "_l7"] = v.shift(1), v.shift(7)
    return X


def backtest(df, group, model, scheme, extra, start, end, refit=7):
    rb.features = lambda d, g: features(d, g, extra)
    try:
        return rb.backtest(df, group, model, scheme, start, end, refit)
    finally:
        rb.features = BASE_FEATURES


if __name__ == "__main__":
    df = rb.load(until="2026-10-31")
    S, E = "2025-06-30", "2026-10-31"
    grid = [(g, m, s, x) for g in ("wx", "mkt") for m in ("ridge", "gbm_d") for s in ("expanding", "ew90")
            for x in ("", "hol", "comp", "hol+comp")]
    out = Parallel(n_jobs=4)(delayed(backtest)(df, *k, S, E) for k in grid)
    P = pd.DataFrame({"|".join(k): v for k, v in zip(grid, out)})
    y = df.y.reindex(P.index)
    P["naive_last"] = df.y.shift(1).reindex(P.index)
    for g in ("wx", "mkt"):
        for s in ("expanding", "ew90"):
            for x in ("", "hol+comp"):
                P[f"{g}|ens|{s}|{x}"] = (P[f"{g}|ridge|{s}|{x}"] + P[f"{g}|gbm_d|{s}|{x}"]) / 2
    ok = y.notna() & P.notna().all(axis=1)
    r = P[ok].sub(y[ok], axis=0).abs().mean().sort_values()
    base = r["naive_last"]
    print(f"n={ok.sum()} post window {S}..{E}")
    print(pd.DataFrame({"MAE": r.round(2), "skill": (1 - r / base).round(3)}).to_string())
    P.to_csv("data/iterate_predictions.csv")
