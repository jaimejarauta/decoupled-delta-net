# Option B exploration (2026-11-14). Development data only: delivery days up to DEV_END. Days after it are the
# planned test period and must not be read here.
import pandas as pd, numpy as np
import i90

DEV_END = "2025-12-31"
BREAK = "2025-04-28"


def hourly(until=DEV_END, units=False):
    um = i90.unit_map()
    p1 = pd.read_csv("data/i90/phase1.csv.gz")
    p1 = p1[(p1.date <= until) & (p1.Sentido == "Subir") & p1.Redespacho.str.startswith("UPOPVP")]
    p1["tech"] = p1["Unidad de Programación"].map(lambda u: um.get(u, "unmapped"))
    sync_units = p1[p1.tech.isin(i90.SYNC)].groupby(["date", "hour", "Unidad de Programación"]).mwh.sum()
    R = p1.groupby(["date", "hour"]).mwh.sum().rename("R")
    Rs = sync_units.groupby(["date", "hour"]).sum().rename("R_sync")
    Rt = p1.pivot_table(index=["date", "hour"], columns="Tipo Restricción", values="mwh", aggfunc="sum").add_prefix("R_")
    pt = pd.read_csv("data/i90/pdbf_tech.csv.gz")
    pt = pt[pt.date <= until].pivot_table(index=["date", "hour"], columns="tech", values="mwh", aggfunc="sum")
    S = pt[[c for c in pt.columns if c in i90.SYNC]].sum(axis=1).rename("S_mkt")
    su = pd.read_csv("data/i90/pdbf_sync_units.csv.gz")
    su = su[su.date <= until].set_index(["date", "hour", "unit"]).mwh
    su.index.names = sync_units.index.names
    post = su.add(sync_units, fill_value=0)
    N_mkt = (su > 1).groupby(["date", "hour"]).sum().rename("N_mkt")
    N_post = (post > 1).groupby(["date", "hour"]).sum().rename("N_post")
    cc = [u for u, t in um.items() if t == "cc"]
    lvl = post.index.get_level_values(2).isin(cc)
    N_cc_post = (post[lvl] > 1).groupby(["date", "hour"]).sum().rename("N_cc_post")
    N_cc_mkt = (su[su.index.get_level_values(2).isin(cc)] > 1).groupby(["date", "hour"]).sum().rename("N_cc_mkt")
    H = pd.concat([R, Rs, S, N_mkt, N_post, N_cc_mkt, N_cc_post, Rt, (-pt["other"]).rename("demand"),
                   (pt.get("unmapped", 0) + pt.get("solar", 0) + pt.get("wind", 0)).rename("ren_proxy")], axis=1)
    H = H.fillna({c: 0 for c in H.columns if c.startswith("R")})
    H["S_post"] = H.S_mkt + H.R_sync
    H = H.reset_index()
    H["period"] = np.where(H.date < "2025-04-28", "pre", np.where(H.date <= "2025-04-30", "gap", "post_dev"))
    return (H, post) if units else H


if __name__ == "__main__":
    H = hourly()
    H.to_csv("data/i90/hourly_dev.csv.gz", index=False, compression="gzip")
    H = H[H.period != "gap"]
    q = lambda s: s.quantile([0.01, 0.05, 0.5]).round(0).tolist()
    for k, g in H.groupby("period"):
        print(k, "hours", len(g), "R>0 share", round((g.R > 0).mean(), 2), "R_sync share of R", round(g.R_sync.sum() / g.R.sum(), 2))
        print("   S_mkt q1/5/50", q(g.S_mkt), " S_post", q(g.S_post), " N_cc_mkt", q(g.N_cc_mkt), " N_cc_post", q(g.N_cc_post))
    g = H[H.period == "post_dev"].copy()
    g["bin"] = pd.cut(g.S_mkt, [0, 6000, 8000, 10000, 12000, 14000, 16000, 20000, 40000])
    print(g.groupby("bin", observed=True)[["R_sync", "S_post", "N_cc_mkt", "N_cc_post"]].mean().round(0).assign(
        n=g.groupby("bin", observed=True).size()).to_string())
    g["cbin"] = pd.cut(g.N_cc_mkt, [-1, 2, 4, 6, 8, 10, 14, 40])
    print(g.groupby("cbin", observed=True)[["R_sync", "N_cc_post"]].mean().round(1).assign(
        n=g.groupby("cbin", observed=True).size()).to_string())
    print("post_dev by month: N_cc_post q1/q5/median, S_post q1")
    g["m"] = g.date.str[:7]
    print(g.groupby("m").agg(ncc_q01=("N_cc_post", lambda s: s.quantile(.01)), ncc_q05=("N_cc_post", lambda s: s.quantile(.05)),
                             ncc_med=("N_cc_post", "median"), S_q01=("S_post", lambda s: s.quantile(.01)),
                             R=("R", "mean")).round(0).to_string())
