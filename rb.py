"""redispatch-break: data build + forecasting benchmark for Spain's Phase-I upward technical-constraint
redispatch across the 2025-04-28 operating-regime change. Question, design and frozen gate: README.md."""
import io
import json
import os
import re
import time
import unicodedata
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("RB_DATA", os.path.join(HERE, "data"))
START = pd.Timestamp("2023-01-01")
BREAK = pd.Timestamp("2025-04-28")      # blackout day
DEV_END = pd.Timestamp("2025-04-27")    # last delivery day any development run may read
ARCH = "https://api.esios.ree.es/archives/{i}/download?date_type=datos&start_date={a}&end_date={b}&locale=es"
WX_VARS = ["shortwave_radiation_previous_day2", "wind_speed_100m_previous_day2", "temperature_2m_previous_day2"]
WX = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&start_date={a}"
      "&end_date={b}&hourly=" + ",".join(WX_VARS) + "&timezone=Europe%2FMadrid")
# Sevilla, Badajoz, Albacete, Zaragoza, Valladolid, Lugo, Burgos, Madrid: main solar and wind regions.
POINTS = [(37.39, -5.99), (38.88, -6.97), (39.00, -1.86), (41.65, -0.89), (41.65, -4.72), (43.01, -7.56),
          (42.34, -3.70), (40.42, -3.70)]
TECH = ["ciclo_combinado", "nuclear", "hidraulica", "eolica", "solar_fotovoltaica", "cogeneracion"]
# Generation concepts of the PDBF table. Their sum is the day's scheduled generation; the demand rows changed
# name in May 2024, so they are not used.
GEN = TECH + ["solar_termica", "otras_renovables", "turbinacion_bombeo", "carbon", "residuos_no_renovables",
              "fuel_gas", "almacenamiento_generacion"]


# ----------------------------------------------------------------------------- download

def fetch(url, path, refresh=False, tries=4):
    """Download to `path` atomically (tmp + fsync + replace). Existing files are kept unless refresh."""
    if os.path.exists(path) and not refresh:
        return path
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-script"})
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read()
            with open(path + ".tmp", "wb") as f:
                f.write(body)
                f.flush()
                os.fsync(f.fileno())
            os.replace(path + ".tmp", path)
            time.sleep(0.3)   # stay polite to the public endpoints
            return path
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def download(end=None):
    """ESIOS archives 19 (totalrpdvpprec, target) and 32 (I3DIA, schedules by technology), one zip per
    month, plus Open-Meteo two-day-ahead forecasts per year. Months that can still change are refetched."""
    end = pd.Timestamp(end or pd.Timestamp.now().normalize())
    os.makedirs(f"{DATA}/raw", exist_ok=True)
    for m in pd.period_range(START, end, freq="M"):
        a, b = m.start_time, min(m.end_time.normalize(), end)
        final = m.end_time < end - pd.Timedelta(days=5)   # I3DIA appears at D+3
        for i in (19, 32):
            fetch(ARCH.format(i=i, a=a.date(), b=b.date()), f"{DATA}/raw/{i}_{m}.zip", refresh=not final)
    lat, lon = (",".join(str(p[k]) for p in POINTS) for k in (0, 1))
    for y in range(2024, end.year + 1):   # the previous-runs archive has no wind/radiation before 2024
        b = min(pd.Timestamp(f"{y}-12-31"), end)
        fetch(WX.format(lat=lat, lon=lon, a=f"{y}-01-01", b=b.date()), f"{DATA}/raw/wx_{y}.json",
              refresh=y >= end.year)


# ----------------------------------------------------------------------------- parsers

def leaves(name, body):
    """Yield (name, bytes) for every file inside a possibly nested zip; a non-zip body is its own leaf."""
    if body[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            for n in z.namelist():
                yield from leaves(n, z.read(n))
    else:
        yield name, body


def slug(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def parse_prec(body):
    """totalrpdvpprec XML -> one row: energy (MWh) and gross payment (EUR) of the PDBF constraint solution.
    Interval quantities are MWh per interval, so the daily energy is the plain sum (checked against I90)."""
    root = ET.fromstring(body)
    ns = root.tag[:root.tag.index("}") + 1]
    out = {"date": pd.Timestamp(root.find(ns + "IdentificacionMensaje").get("v").split("_")[-1]),
           "pub": root.find(ns + "FechaHoraMensaje").get("v"), "up_mwh": 0.0, "down_mwh": 0.0, "up_eur": 0.0}
    for s in root.iter(ns + "SeriesTemporales"):
        for it in s.iter(ns + "Intervalo"):
            g = {c.tag[len(ns):]: float(c.get("v")) for c in it}
            out["up_mwh"] += g.get("CtdSub", 0.0)
            out["down_mwh"] += g.get("CtdBaj", 0.0)
            out["up_eur"] += g.get("CtdSub", 0.0) * g.get("PrecioSub", 0.0)
    return out


def sheet_rows(sh):
    """(labels, total) per data row of an ESIOS report sheet: the leading non-empty text cells, then the
    first number, which is the daily 'Total' column in every table used here."""
    for r in range(sh.nrows):
        labs = []
        for v in sh.row_values(r):
            if isinstance(v, str):
                if v.strip():
                    labs.append(v.strip())
            else:
                if labs:
                    yield labs, float(v)
                break


class CsvSheet:
    """REE serves I3DIA as one semicolon CSV per sheet from October 2026 (decimal comma, Latin-1). This gives
    such a file the two members of an xlrd sheet that the code uses."""

    def __init__(self, body):
        num = re.compile(r"^-?\d+(,\d+)?$")
        self.rows = [[float(c.replace(",", ".")) if num.match(c) else c for c in line.split(";")]
                     for line in body.decode("latin-1").splitlines()]
        self.nrows = len(self.rows)

    def row_values(self, r):
        return self.rows[r]


def fold_sheet(out, sh):
    """Add one report sheet to the day's row `out` if it is one of the three tables used: PDBF by technology,
    PDVP by technology, or PDBF constraint redispatch by direction, technology and restriction type.
    Sheets are recognised by content, not by number."""
    top = {str(v).strip() for r in range(min(6, sh.nrows)) for v in sh.row_values(r)}
    if {"Redespacho", "Concepto", "Tipo Restricción"} <= top:
        kind = "rd"
    elif "Concepto" in top and "PDBF" in top:
        kind = "pdbf"
    elif "Concepto" in top and "PDVP" in top:
        kind = "pdvp"
    else:
        return
    for labs, v in sheet_rows(sh):
        if labs[0] == "Dia":
            out["date"] = pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(v))   # Excel serial date
        elif labs[0] in ("Concepto", "Redespacho", "Indicadores"):
            continue
        elif kind == "rd" and len(labs) >= 3:
            code, sentido, tech = labs[0], labs[1], slug(labs[2])
            tipo = slug(labs[3]) if len(labs) > 3 else "na"
            # P.O. 3.2 codes. From October 2025 the real-time sheet has the same columns, so the code decides.
            if code.startswith("UPOPVP") and sentido == "Subir":
                keys = ["rd_up", f"rdup_type_{tipo}", f"rdup_tech_{tech}"]   # Phase I security, upward
            elif code.startswith("UPLPVP"):
                keys = ["rd_dn"]                                    # Phase I, limited downward
            elif code.startswith("ECO"):
                keys = ["rd_eco"]                                   # Phase II rebalancing
            elif code.startswith("UPOTR") and sentido == "Subir":
                keys = ["rt_up"]                                    # real-time constraints, upward
            else:
                continue
            for k in keys:
                out[k] = out.get(k, 0.0) + v
        elif kind in ("pdbf", "pdvp"):
            k = f"{kind}_{slug(labs[0])}"
            out[k] = out.get(k, 0.0) + v


def parse_i3(body, name=""):
    """One I3DIA file -> a (partial) row for its delivery day. A workbook holds every sheet of the day; a CSV
    holds one sheet and takes the day from its file name, so rows of the same day are merged by the caller."""
    out = {}
    if body[:4] == b"\xd0\xcf\x11\xe0":
        import xlrd
        wb = xlrd.open_workbook(file_contents=body, on_demand=True)
        for sheet in wb.sheet_names():
            fold_sheet(out, wb.sheet_by_name(sheet))
            wb.unload_sheet(sheet)
    else:
        out["date"] = pd.Timestamp(re.search(r"_(\d{8})\.csv$", name).group(1))
        fold_sheet(out, CsvSheet(body))
    return out


def parse_wx(body):
    """Open-Meteo previous-runs JSON (one entry per point) -> daily national means of two-day-ahead forecasts."""
    pts = json.loads(body)
    pts = pts if isinstance(pts, list) else [pts]
    h = sum(pd.DataFrame(p["hourly"]).set_index("time")[WX_VARS].astype(float) for p in pts) / len(pts)
    h.index = pd.to_datetime(h.index)
    g = h.groupby(h.index.normalize())
    d = pd.DataFrame({"wx_rad": g[WX_VARS[0]].sum(min_count=20), "wx_wind": g[WX_VARS[1]].mean(),
                      "wx_temp": g[WX_VARS[2]].mean()})
    d[g[WX_VARS[1]].count() < 20] = np.nan
    return d


def build():
    """Parse everything under data/raw into data/daily.csv (one row per delivery day). Parsed months are cached."""
    os.makedirs(f"{DATA}/cache", exist_ok=True)
    frames, skipped = {19: [], 32: []}, []
    for fn in sorted(os.listdir(f"{DATA}/raw")):
        m = re.match(r"(19|32)_(\d{4}-\d{2})\.zip$", fn)
        if not m:
            continue
        raw, cache = f"{DATA}/raw/{fn}", f"{DATA}/cache/{fn[:-4]}.csv"
        if not os.path.exists(cache) or os.path.getmtime(cache) < os.path.getmtime(raw):
            rows = [{"date": pd.NaT}]
            with open(raw, "rb") as f:
                for name, b in leaves(fn, f.read()):
                    try:
                        rows.append(parse_prec(b) if m.group(1) == "19" else parse_i3(b, name))
                    except Exception as e:
                        skipped.append(f"{fn}:{name}: {type(e).__name__}: {str(e)[:80]}")
            # one row per day: a day's CSV sheets arrive as separate partial rows
            pd.DataFrame(rows).dropna(subset=["date"]).groupby("date", as_index=False).first().to_csv(
                cache + ".tmp", index=False)
            os.replace(cache + ".tmp", cache)
        frames[int(m.group(1))].append(pd.read_csv(cache, parse_dates=["date"]))
    parts = [pd.concat(v).drop_duplicates("date", keep="last").set_index("date") for v in frames.values()]
    wx = [parse_wx(open(f"{DATA}/raw/{fn}", "rb").read()) for fn in sorted(os.listdir(f"{DATA}/raw"))
          if fn.startswith("wx_")]
    if wx:
        parts.append(pd.concat(wx).pipe(lambda d: d[~d.index.duplicated(keep="last")]))
    df = pd.concat(parts, axis=1).sort_index()
    df = df.reindex(pd.date_range(START, df.index.max()))
    # Target: Phase-I (security) upward energy, GWh per delivery day. The XML total also contains Phase-II
    # upward rebalancing, so it is kept only as a cross-check and for the publication timestamp.
    df["y"] = df["rd_up"] / 1000.0
    gen = [c for c in df if c.startswith("pdbf_") and (c[5:] in GEN or c.startswith("pdbf_hibrida"))]
    have = df["pdbf_hidraulica"].notna()            # the PDBF table was read; a technology with no row scheduled 0
    df.loc[have, gen] = df.loc[have, gen].fillna(0.0)
    df["pdbf_gen"] = df[gen].sum(axis=1, min_count=1)
    df.index.name = "date"
    if skipped:   # unreadable days stay missing in the table and count against the coverage check
        print(f"{len(skipped)} unreadable files skipped:", *skipped[:10], sep="\n  ")
    df.to_csv(f"{DATA}/daily.csv.tmp")
    os.replace(f"{DATA}/daily.csv.tmp", f"{DATA}/daily.csv")
    return df


def load(until=None):
    """Daily table. `until` truncates it, so development code cannot read post-break rows."""
    df = pd.read_csv(f"{DATA}/daily.csv", parse_dates=["date"], index_col="date")
    return df if until is None else df[df.index <= pd.Timestamp(until)]


REDATA = ("https://apidatos.ree.es/es/datos/mercados/energia-restricciones?start_date={y}-01-01T00:00"
          "&end_date={y}-12-31T23:59&time_trunc=month")


def redata_monthly(years):
    """REE's own monthly 'Energía a subir restricciones técnicas al PDBF (fase I)' (REData series 701), MWh."""
    out = {}
    for y in years:
        req = urllib.request.Request(REDATA.format(y=y), headers={"User-Agent": "research-script"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.load(r)
        for s in d["included"]:
            if str(s["id"]) == "701":
                for v in s["attributes"]["values"]:
                    out[pd.Timestamp(v["datetime"][:10])] = float(v["value"])
    return pd.Series(out).sort_index()


def integrity(df, ref=None):
    """Data checks used as validity conditions. `ref` is an independent monthly series (redata_monthly):
    complete months of the parsed target must match it. The XML total (Phase I + II upward) bounds the target
    from above, and its timestamp shows the solution is public before the delivery day starts (UTC)."""
    both = df[["up_mwh", "rd_up"]].dropna()
    pub = pd.to_datetime(df["pub"].dropna(), utc=True).dt.tz_localize(None)
    out = {"days": int(len(df)), "target_missing": int(df.y.isna().sum()), "n_both_sources": int(len(both)),
           "xml_within_1pct_frac": float(((both.up_mwh - both.rd_up).abs() <= 0.01 * both.up_mwh + 1).mean()),
           "target_le_xml_frac": float((both.rd_up <= both.up_mwh * 1.001 + 1).mean()),
           "published_before_delivery_frac": float((pub < pub.index).mean())}
    if ref is not None:
        m = df.rd_up.groupby(df.index.to_period("M")).agg(["sum", "count"])
        m = m[m["count"] == m.index.days_in_month]["sum"]
        r = ref.copy()
        r.index = r.index.to_period("M")
        rel = ((m - r.reindex(m.index)) / r.reindex(m.index)).dropna()
        out.update(redata_months=int(len(rel)), redata_within_2pct_frac=float((rel.abs() <= 0.02).mean()),
                   redata_max_abs_rel=float(rel.abs().max()))
    return out


# ----------------------------------------------------------------------------- features

def features(df, group):
    """Feature table for delivery day D. Row D uses only information public at the issue time:
    'lags' and 'wx': D-1 10:00 (before the day-ahead gate closes): calendar, target and PDBF up to D-1,
                     and for 'wx' the weather forecast for D made two days before;
    'mkt': D-1 13:30 (after the PDBF is published, before the constraint solution): adds the PDBF of D."""
    X = pd.DataFrame(index=df.index)
    doy = 2 * np.pi * df.index.day_of_year / 365.25
    X["dow"], X["wkend"] = df.index.day_of_week, (df.index.day_of_week >= 5).astype(float)
    X["doy_s"], X["doy_c"] = np.sin(doy), np.cos(doy)
    for lag in (1, 2, 7):
        X[f"y_l{lag}"] = df.y.shift(lag)
    X["y_m7"] = df.y.shift(1).rolling(7).mean()
    gen = df["pdbf_gen"]
    for t in TECH:
        X[f"{t}_l1"] = df[f"pdbf_{t}"].shift(1)
    X["gen_l1"] = gen.shift(1)
    if group in ("wx", "mkt"):
        for c in ("wx_rad", "wx_wind", "wx_temp"):
            X[c] = df[c]
    if group == "mkt":
        for t in TECH:
            X[t] = df[f"pdbf_{t}"]
        X["gen"] = gen
        X["sync_share"] = (df.pdbf_ciclo_combinado + df.pdbf_nuclear + df.pdbf_hidraulica) / gen
    return X


def naive(df):
    return pd.DataFrame({"naive_last": df.y.shift(1), "naive_week": df.y.shift(7),
                         "naive_m7": df.y.shift(1).rolling(7).mean()})


# ----------------------------------------------------------------------------- models

def weights(dates, scheme, now):
    """Training weights by sample age: rollN keeps the last N days, ewH halves the weight every H days."""
    age = np.asarray((now - dates).days, dtype=float)
    if scheme.startswith("roll"):
        return (age <= int(scheme[4:])).astype(float)
    if scheme.startswith("ew"):
        return 0.5 ** (age / int(scheme[2:]))
    return np.ones(len(age))


def fit_predict(model, Xtr, ytr, w, Xte):
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import Ridge
    if model == "ridge":
        med = Xtr.median().fillna(0.0)
        mu, sd = Xtr.fillna(med).mean(), Xtr.fillna(med).std().replace(0, 1).fillna(1)
        z = lambda X: (X.fillna(med) - mu) / sd
        return Ridge(alpha=1.0).fit(z(Xtr), ytr, sample_weight=w).predict(z(Xte))
    keep = Xtr.columns[Xtr.notna().any()]            # e.g. weather forecasts do not exist before 2024
    Xtr, Xte = Xtr[keep], Xte[keep]
    gbm = HistGradientBoostingRegressor(loss="absolute_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=8,
                                        min_samples_leaf=10, l2_regularization=1.0, random_state=0)
    if model == "gbm":
        return gbm.fit(Xtr, ytr, sample_weight=w).predict(Xte)
    if model == "gbm_d":   # trees cannot extrapolate a level shift, so predict the change from yesterday
        return gbm.fit(Xtr, ytr - Xtr["y_l1"], sample_weight=w).predict(Xte) + Xte["y_l1"].to_numpy()
    raise ValueError(model)


def backtest(df, group, model, scheme, start, end, refit=7):
    """Rolling-origin forecasts for every delivery day in [start, end]. The model is refitted every `refit`
    days on days strictly before the block; lag features still update daily."""
    X, y = features(df, group), df.y
    if scheme == "dummy":
        X["post"] = (X.index > BREAK).astype(float)
    days = pd.date_range(start, min(pd.Timestamp(end), df.index.max()))
    pred = pd.Series(np.nan, index=days)
    for i in range(0, len(days), refit):
        blk = days[i:i + refit]
        tr = (X.index < blk[0]) & y.notna().to_numpy() & X["y_m7"].notna().to_numpy()
        w = weights(X.index[tr], scheme, blk[0])
        keep = w > 0
        te = X.loc[blk]
        ok = te["y_l1"].notna().to_numpy()
        if keep.sum() >= 60 and ok.any():
            pred[blk[ok]] = fit_predict(model, X[tr][keep], y[tr][keep], w[keep], te[ok])
    return pred


def online_select(P, y, cols, window=56, min_periods=14):
    """Each day follow the column with the lowest mean absolute error over the previous `window` days."""
    E = P[cols].sub(y, axis=0).abs()
    R = E.shift(1).rolling(window, min_periods=min_periods).mean()
    best = R[R.notna().any(axis=1)].idxmin(axis=1).reindex(P.index)   # no forecast until errors have been seen
    out = pd.Series(np.nan, index=P.index)
    for c in cols:
        out[best == c] = P.loc[best == c, c]
    return out


# ----------------------------------------------------------------------------- metrics

def mae(y, p):
    return float((y - p).abs().mean())


def skill(y, p, p0):
    return 1.0 - mae(y, p) / mae(y, p0)


def _blocks(n, block, rng):
    k = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), size=k)
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts])[:n]


def boot_ci(stat, n, block=7, reps=2000, seed=0):
    """95% moving-block bootstrap interval of stat(index array) over n days."""
    rng = np.random.default_rng(seed)
    v = [stat(_blocks(n, block, rng)) for _ in range(reps)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def dm_test(y, p, p0, lag=7):
    """Diebold-Mariano on absolute errors with a Newey-West variance. Negative t: p beats p0."""
    from math import erf, sqrt
    d = ((y - p).abs() - (y - p0).abs()).dropna().to_numpy()
    n, dc = len(d), d - d.mean()
    var = dc @ dc / n + 2 * sum((1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n for k in range(1, lag + 1))
    t = d.mean() / np.sqrt(max(var, 1e-12) / n)
    return float(t), float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def recovery_day(y, p, p0, t0, win=14, hold=14, horizon=365):
    """Days after t0 until the trailing-`win`-day MAE of p is no worse than that of p0 for `hold` days in a
    row; `horizon` if that never happens inside the horizon."""
    idx = pd.date_range(t0, periods=horizon)
    idx = idx[idx <= y.index.max()]
    good = ((y - p).abs().rolling(win).mean() <= (y - p0).abs().rolling(win).mean()).reindex(idx).fillna(False)
    run = good.rolling(hold).sum()
    hit = run[run >= hold]
    return int((hit.index[0] - t0).days - hold + 1) if len(hit) else horizon


def counterfactual(df, train_end, test_start, test_end, model="gbm"):
    """Old-regime redispatch for [test_start, test_end]: a model of the target from the day's market schedule
    and calendar only (no target lags, which would carry the regime), trained on days up to train_end."""
    X = features(df, "mkt")[["dow", "wkend", "doy_s", "doy_c", "gen", "sync_share"] + TECH]
    tr = (X.index <= pd.Timestamp(train_end)) & df.y.notna().to_numpy() & X["gen"].notna().to_numpy()
    te = ((X.index >= pd.Timestamp(test_start)) & (X.index <= pd.Timestamp(test_end))
          & df.y.notna().to_numpy() & X["gen"].notna().to_numpy())
    p = fit_predict(model, X[tr], df.y[tr], np.ones(int(tr.sum())), X[te])
    a = df.y[te].to_numpy()
    return {"n": int(te.sum()), "actual_gwh": float(a.sum()), "counterfactual_gwh": float(p.sum()),
            "ratio": float(a.sum() / p.sum()),
            "ratio_ci": boot_ci(lambda i: a[i].sum() / p[i].sum(), len(a))}


def dump(obj, path):
    """Atomic JSON; numpy and pandas values are converted, so a stray np.bool_ cannot tear the file."""
    def conv(o):
        return o.tolist() if hasattr(o, "tolist") else str(o)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1, default=conv)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)
