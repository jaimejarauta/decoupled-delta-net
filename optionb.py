"""Option B gate (README "Option B", frozen 2026-11-14).

dev:  reproduce the development numbers on delivery days <= 2025-12-31 (the 2026 test window is not read).
test: one shot on 2026-01-01..2026-08-16. Writes data/optionb_verdict.json. Do not rerun after the first run.
"""
import json, sys
import numpy as np, pandas as pd
import explore_b, i90
from zones import ZONE

REGIONS = sorted(set(ZONE.values()))


def metrics(H, post, a, b):
    h = H[(H.date >= a) & (H.date <= b)]
    n, m = h.N_cc_post.fillna(0), h.N_cc_mkt.fillna(0)
    k = m <= 10
    slope = float(np.polyfit(m[k], n[k], 1)[0])
    d = post[post > 1].reset_index()
    d.columns = ["date", "hour", "unit", "mwh"]
    d = d[(d.date >= a) & (d.date <= b)]
    d["z"] = d.unit.map(ZONE)
    cc = d[d.unit.map(i90.unit_map()).eq("cc")]
    zoned_share = float(cc.mwh[cc.z.notna()].sum() / cc.mwh.sum())
    cover = (d.dropna(subset=["z"]).drop_duplicates(["z", "date", "hour"]).groupby("z").size()
             .reindex(REGIONS, fill_value=0) / len(h))
    return dict(hours=len(h), slope=round(slope, 3), median=float(n.median()), q01=float(n.quantile(.01)),
                cover={z: round(float(c), 3) for z, c in cover.items()}, regions95=int((cover >= 0.95).sum()),
                zoned_cc_share=round(zoned_share, 3))


def validity(a, b):
    """V1 coverage, V2 reconstruction of daily y, V3 technology mapping of Phase-I upward energy."""
    um = i90.unit_map()
    p1 = pd.read_csv("data/i90/phase1.csv.gz")
    p1 = p1[(p1.date >= a) & (p1.date <= b) & (p1.Sentido == "Subir") & p1.Redespacho.str.startswith("UPOPVP")]
    days = pd.date_range(a, b).strftime("%Y-%m-%d")
    rec = p1.groupby("date").mwh.sum()
    y = pd.read_csv("data/daily.csv").set_index("date").rd_up.reindex(rec.index)  # target y in MWh
    mapped = p1.mwh[p1["Unidad de Programación"].isin(um)].sum() / p1.mwh.sum()
    return dict(days=len(days), days_with_data=int(rec.index.isin(days).sum()),
                v1=bool(rec.index.isin(days).sum() >= 0.95 * len(days)),
                share_within_1pct=round(float(((rec - y).abs() <= 0.01 * y).mean()), 3),
                v2=bool(((rec - y).abs() <= 0.01 * y).mean() >= 0.95),
                mapped_share=round(float(mapped), 3), v3=bool(mapped >= 0.85))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "dev":
        H, post = explore_b.hourly(units=True)
        print("pre  2024-01-01..2025-04-27", metrics(H, post, "2024-01-01", "2025-04-27"))
        print("post 2025-05-01..2025-12-31", metrics(H, post, "2025-05-01", "2025-12-31"))
        print("2024 summer window", metrics(H, post, "2024-07-16", "2024-08-16"))
        print("2025 summer window", metrics(H, post, "2025-07-16", "2025-08-16"))
        print("validity dev post", validity("2025-05-01", "2025-12-31"))
    elif mode == "test":
        H, post = explore_b.hourly(until="2026-08-16", units=True)
        T = metrics(H, post, "2026-01-01", "2026-08-16")
        S26 = metrics(H, post, "2026-07-16", "2026-08-16")
        S25 = metrics(H, post, "2025-07-16", "2025-08-16")
        V = validity("2026-01-01", "2026-08-16")
        t1 = T["slope"] <= 0.35 and T["median"] >= 14
        t2 = T["regions95"] >= 6
        t3 = S26["median"] <= S25["median"] - 2
        ok = V["v1"] and V["v2"] and V["v3"]
        verdict = "INVALID" if not ok else {3: "PASS", 0: "FAIL"}.get(t1 + t2 + t3, "MIXED")
        h = H[(H.date >= "2026-01-01") & (H.date <= "2026-08-16")].copy()
        monthly = h.assign(m=h.date.str[:7]).groupby("m").N_cc_post.agg(["median", lambda s: s.quantile(.01)])
        out = dict(verdict=verdict, T1=t1, T2=t2, T3=t3, validity=V, test=T, summer2026=S26, summer2025=S25,
                   monthly_median_q01={m: [float(r.iloc[0]), float(r.iloc[1])] for m, r in monthly.iterrows()})
        json.dump(out, open("data/optionb_verdict.json", "w"), indent=1)
        print(json.dumps(out, indent=1))
