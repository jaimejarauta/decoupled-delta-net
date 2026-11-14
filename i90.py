"""Unit-level Phase-I redispatch from ESIOS archive 34 (I90DIA, published 90 days after delivery).

uv run --no-project --python 3.12 --with-requirements requirements.txt python i90.py fetch 2024-01-01 2026-08-31
uv run --no-project --python 3.12 --with-requirements requirements.txt python i90.py parse

fetch: one zip per delivery day into data/i90/raw/ (resumable; days not yet published are skipped).
parse: sheet I90DIA03 (Phase-I solution of the PDBF technical constraints) to data/i90/phase1.csv.gz, one row per
(date, hour, unit, restriction type, redispatch code, direction) with MWh. Older days come as one .xls workbook with
hourly columns, newer ones as CSV files with quarter-hour columns; both are reduced to hours.
"""
import concurrent.futures as cf, datetime as dt, io, os, sys, time, urllib.request, zipfile
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "i90", "raw")
OUT = os.path.join(HERE, "data", "i90", "phase1.csv.gz")
URL = ("https://api.esios.ree.es/archives/34/download?date_type=datos&start_date={d}T00:00:00"
       "&end_date={d}T23:59:59&locale=es")
KEYS = ["Redespacho", "Sentido", "Unidad de Programación", "Tipo Restricción"]


def fetch_day(d):
    path = os.path.join(RAW, f"{d}.zip")
    if os.path.exists(path):
        return d, "have"
    for k in range(4):
        try:
            req = urllib.request.Request(URL.format(d=d), headers={"User-Agent": "research-script"})
            with urllib.request.urlopen(req, timeout=120) as r:
                body = r.read()
            if not body:
                return d, "empty"
            zipfile.ZipFile(io.BytesIO(body)).testzip()
            with open(path + ".tmp", "wb") as f:
                f.write(body)
            os.replace(path + ".tmp", path)
            return d, "ok"
        except Exception as e:
            err = repr(e)[:80]
            time.sleep(20 * (k + 1))
    return d, "fail " + err


def fetch(a, b, workers=2):
    os.makedirs(RAW, exist_ok=True)
    days = [str(x.date()) for x in pd.date_range(a, b)]
    with cf.ThreadPoolExecutor(workers) as ex:
        for i, (d, s) in enumerate(ex.map(fetch_day, days)):
            if s != "have":
                print(d, s, flush=True)


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                         errors="coerce")


def parse_zip(path):
    """Long table (date, hour, keys..., mwh) of I90DIA03 for one delivery day."""
    d = os.path.basename(path)[:10]
    z = zipfile.ZipFile(path)
    names = z.namelist()
    csv = [n for n in names if n.startswith("I90DIA03") and n.endswith(".csv")]
    if csv:
        raw = z.read(csv[0])
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")  # some 2026 files are not UTF-8
        t = pd.read_csv(io.StringIO(text), sep=";", skiprows=2, dtype=str)
        q = [c for c in t.columns if c.isdigit()]
        v = t[q].apply(_num).fillna(0.0)
        v.columns = [(int(c) - 1) // 4 for c in q]          # quarter hour -> hour (values are MWh per quarter)
        v = v.T.groupby(level=0).sum().T
    else:
        import xlrd
        xls = [n for n in names if n.lower().endswith(".xls")][0]
        s = xlrd.open_workbook(file_contents=z.read(xls), on_demand=True).sheet_by_name("I90DIA03")
        hdr = [str(c.value) for c in s.row(2)]
        t = pd.DataFrame([[c.value for c in s.row(i)] for i in range(3, s.nrows)], columns=hdr)
        hours = hdr[hdr.index("Total") + 1:]
        v = t[hours].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        v.columns = range(len(hours))
    long = v.assign(**{k: t[k].values for k in KEYS}).melt(id_vars=KEYS, var_name="hour", value_name="mwh")
    long = long[long.mwh != 0]
    long.insert(0, "date", d)
    return long


def parse():
    files = sorted(f for f in os.listdir(RAW) if f.endswith(".zip"))
    parts = []
    for i, f in enumerate(files):
        try:
            parts.append(parse_zip(os.path.join(RAW, f)))
        except Exception as e:
            print(f, "parse failed", repr(e)[:120], flush=True)
        if i % 100 == 0:
            print(i, "of", len(files), flush=True)
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(OUT + ".tmp", index=False, compression="gzip")
    os.replace(OUT + ".tmp", OUT)
    print(len(df), "rows,", df.date.nunique(), "days ->", OUT)


SYNC = {"cc", "nuclear", "coal", "gas", "hydro"}   # synchronous machines that can control voltage


def tech_class(t):
    t = str(t).lower()
    for k, c in [("ciclo combinado", "cc"), ("nuclear", "nuclear"), ("hulla", "coal"), ("carb", "coal"),
                 ("lignito", "coal"), ("fuel", "gas"), ("gas", "gas"), ("bombeo", "pumped"),
                 ("hidr", "hydro"), ("fotovolt", "solar"), ("solar t", "solar_th"), ("eólica", "wind"),
                 ("eolica", "wind"), ("almacenamiento", "storage"), ("import", "import"), ("export", "export"),
                 ("térmica", "other_thermal"), ("termica", "other_thermal")]:
        if k in t:
            return c
    return "other"


def unit_map():
    """REE programming unit -> technology class, from OMIE's public list of offering units (downloaded 2026-11-14).
    REE codes sometimes drop OMIE's 'R' (represented) or 'G' suffix; two units are mapped by hand."""
    u = pd.read_csv(os.path.join(HERE, "data", "units_omie.csv"))
    m = {c: tech_class(t) for c, t in zip(u.code, u.tech)}
    for c in list(m):
        if c[-1] in "RG" and c[:-1] not in m:
            m[c[:-1]] = m[c]
    m.update({"CAMGI20": "cc", "TMOVIAS": "other_thermal", "TMOVIAL": "other_thermal"})  # by hand, from OMIE names
    return m


def _sheet(z, name, skip_csv):
    """(frame, hour columns) of one I90 sheet, hours 0..23, for either file format."""
    names = z.namelist()
    csv = [n for n in names if n.startswith(name) and n.endswith(".csv")]
    if csv:
        raw = z.read(csv[0])
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
        t = pd.read_csv(io.StringIO(text), sep=";", skiprows=skip_csv, dtype=str)
        q = [c for c in t.columns if c.isdigit()]
        v = t[q].apply(_num).fillna(0.0)
        v.columns = [(int(c) - 1) // 4 for c in q]
        return t, v.T.groupby(level=0).sum().T
    import xlrd
    xls = [n for n in names if n.lower().endswith(".xls")][0]
    s = xlrd.open_workbook(file_contents=z.read(xls), on_demand=True).sheet_by_name(name)
    hrow = next(i for i in range(8) if str(s.cell_value(i, 0)).startswith(("Unidad", "Redespacho")))
    hdr = [str(c.value) for c in s.row(hrow)]
    t = pd.DataFrame([[c.value for c in s.row(i)] for i in range(hrow + 1, s.nrows)], columns=hdr)
    hours = hdr[hdr.index("Total") + 1:]
    v = t[hours].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    v.columns = range(len(hours))
    return t, v


def parse_pdbf(path, umap):
    """Hourly PDBF (day-ahead market + bilateral schedule) per technology class, and per unit for synchronous ones."""
    d = os.path.basename(path)[:10]
    t, v = _sheet(zipfile.ZipFile(path), "I90DIA26", 3)
    v["unit"] = t["Unidad de Programación"].values
    v = v.groupby("unit").sum()
    tech = v.index.map(lambda u: umap.get(u, "unmapped"))
    by_tech = v.groupby(tech).sum().stack().rename("mwh").reset_index()
    by_tech.columns = ["tech", "hour", "mwh"]
    sync = v[tech.isin(SYNC)].stack().rename("mwh").reset_index()
    sync.columns = ["unit", "hour", "mwh"]
    sync = sync[sync.mwh != 0]
    return by_tech.assign(date=d), sync.assign(date=d)


def parse_pdbf_all():
    umap = unit_map()
    files = sorted(f for f in os.listdir(RAW) if f.endswith(".zip"))
    a, b = [], []
    for i, f in enumerate(files):
        try:
            x, y = parse_pdbf(os.path.join(RAW, f), umap)
            a.append(x); b.append(y)
        except Exception as e:
            print(f, "pdbf parse failed", repr(e)[:120], flush=True)
        if i % 100 == 0:
            print(i, "of", len(files), flush=True)
    base = os.path.join(HERE, "data", "i90")
    pd.concat(a).to_csv(os.path.join(base, "pdbf_tech.csv.gz"), index=False, compression="gzip")
    pd.concat(b).to_csv(os.path.join(base, "pdbf_sync_units.csv.gz"), index=False, compression="gzip")
    print("done", len(a), "days")


def check(df_long):
    """Daily Phase-I upward sum per day; compare with the target y of data/daily.csv."""
    up = df_long[(df_long.Sentido == "Subir") & df_long.Redespacho.str.startswith("UPOPVP")]
    return up.groupby("date").mwh.sum() / 1e3


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        fetch(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "parse":
        parse()
    elif sys.argv[1] == "pdbf":
        parse_pdbf_all()
