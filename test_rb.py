"""pytest for redispatch-break. No network. The last test also checks the real table if data/daily.csv exists."""
import io
import json
import os
import zipfile

import numpy as np
import pandas as pd
import pytest

import rb
import run_all

XML = """<TotalRPdvpPrec xmlns="urn:sios.ree.es:totalrpdvpprec:1:0"><IdentificacionMensaje v="totalrpdvpprec_20250610"/>
<FechaHoraMensaje v="2025-06-09T12:38:50Z"/><SeriesTemporales><TipoRedespacho v="21"/><Periodo>
<Intervalo><Pos v="1"/><CtdSub v="10.0"/><PrecioSub v="100.0"/></Intervalo>
<Intervalo><Pos v="2"/><CtdSub v="30.0"/><PrecioSub v="200.0"/></Intervalo></Periodo></SeriesTemporales>
<SeriesTemporales><TipoRedespacho v="43"/><Periodo><Intervalo><Pos v="1"/><CtdBaj v="7.5"/><PrecioBaj v="50"/></Intervalo>
</Periodo></SeriesTemporales></TotalRPdvpPrec>"""


def synth(n=1000, shift_at=None, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n)
    dem = 600 + 50 * np.sin(2 * np.pi * idx.day_of_year / 365.25) + rng.normal(0, 10, n)
    ccgt = np.clip(40 + rng.normal(0, 15, n), 0, None)
    y = 30 - 0.4 * ccgt + 3 * (idx.day_of_week >= 5) + rng.normal(0, 2, n)
    if shift_at:
        y = y + 40 * (idx >= shift_at)
    df = pd.DataFrame({"y": y, "pdbf_gen": dem * 1000, "pdbf_ciclo_combinado": ccgt * 1000,
                       "wx_rad": rng.normal(5000, 500, n), "wx_wind": rng.normal(20, 5, n),
                       "wx_temp": rng.normal(15, 5, n)}, index=idx)
    for t in rb.TECH[1:]:
        df[f"pdbf_{t}"] = rng.normal(100, 10, n) * 1000
    return df


def test_parse_prec():
    r = rb.parse_prec(XML.encode())
    assert r["date"] == pd.Timestamp("2025-06-10") and r["up_mwh"] == 40.0 and r["down_mwh"] == 7.5
    assert r["up_eur"] == 10 * 100 + 30 * 200


def test_leaves_nested_zip_and_plain():
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("a.xml", "A")
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("day.zip", inner.getvalue())
        z.writestr("b.xml", "B")
    assert sorted(rb.leaves("m.zip", outer.getvalue())) == [("a.xml", b"A"), ("b.xml", b"B")]
    assert list(rb.leaves("one.xml", b"<x/>")) == [("one.xml", b"<x/>")]


def test_sheet_rows_and_slug():
    class Sheet:
        rows = [["", "", "Dia", 45818.0], ["Concepto", "Tipo Transacción", "", "Total"],
                ["Hidráulica", "Mercado", "", 77370.0, 1.0], ["", "", ""], [5.0, "x"]]
        nrows = len(rows)

        def row_values(self, r):
            return self.rows[r]
    assert list(rb.sheet_rows(Sheet())) == [(["Dia"], 45818.0), (["Hidráulica", "Mercado"], 77370.0)]
    assert rb.slug("Híbrida Renovable - Térmica ") == "hibrida_renovable_termica"


def test_csv_sheets_parse_like_workbook_sheets():
    rd = ("\n;;;;15-oct-2026\nRedespacho;Sentido;Concepto;Tipo Restricción;Total;1;2\n"
          "UPOPVPV;Subir;Ciclo Combinado;RSI;51297,000;487,300;487,300\nUPOPVPV;Subir;Fuel-Gas;RTD;8500,000;75,000\n"
          "UPLPVPV;Bajar;Eólica;RTD;-292,300;-1,0\nECO;Bajar;Hidráulica;N/A;-16255,600;0,000\n"
          "UPOTROR;Subir;Ciclo Combinado;SOT;2279,000;1,0\nDESCOM2;Subir;Eólica;N/A;522,000;1,0\n").encode("latin-1")
    r = rb.parse_i3(rd, "I3DIA03_20261015.csv")
    assert r["date"] == pd.Timestamp("2026-10-15") and r["rd_up"] == 59797.0 and r["rdup_type_rsi"] == 51297.0
    assert r["rdup_tech_ciclo_combinado"] == 51297.0 and r["rd_dn"] == -292.3 and r["rd_eco"] == -16255.6
    assert r["rt_up"] == 2279.0      # real-time rows never enter the Phase-I target
    pdbf = (";;PDBF;PDBF\nConcepto;Tipo Transacción;Total;1\nHidráulica;Mercado;51656,900;721,200\n"
            "Hidráulica;Bilateral;19961,900;171,100\nAdquisición de Energía;Mercado;-369527,200;1\n").encode("latin-1")
    p = rb.parse_i3(pdbf, "I3DIA14_20261015.csv")
    assert p["pdbf_hidraulica"] == pytest.approx(71618.8) and p["pdbf_adquisicion_de_energia"] == -369527.2
    assert rb.parse_i3(b"Sin contenido\n", "I3DIA04_20261015.csv") == {"date": pd.Timestamp("2026-10-15")}


def test_features_use_only_past_information():
    df, d = synth(), pd.Timestamp("2024-06-01")
    for group in ("lags", "wx", "mkt"):
        base = rb.features(df, group)
        x = df.copy()
        x.loc[d:, "y"] += 1000                       # the target of day d and later must never be visible
        if group != "mkt":
            x.loc[d:, [c for c in x if c.startswith("pdbf_")]] *= 3   # nor the market schedule, before it clears
        pd.testing.assert_frame_equal(base.loc[:d], rb.features(x, group).loc[:d])
    with pytest.raises(AssertionError):              # the check has teeth: 'mkt' does read the PDBF of day d
        x = df.copy()
        x.loc[d:, "pdbf_ciclo_combinado"] *= 3
        pd.testing.assert_frame_equal(rb.features(df, "mkt").loc[:d], rb.features(x, "mkt").loc[:d])


def test_backtest_has_no_look_ahead_and_beats_naive_with_market_features():
    df, k = synth(), pd.Timestamp("2024-08-15")
    p = rb.backtest(df, "mkt", "ridge", "expanding", "2024-06-01", "2024-09-30")
    x = df.copy()
    x.loc[k:, "y"] += 500
    q = rb.backtest(x, "mkt", "ridge", "expanding", "2024-06-01", "2024-09-30")
    pd.testing.assert_series_equal(p[:k], q[:k])
    assert not p[k + pd.Timedelta(days=8):].equals(q[k + pd.Timedelta(days=8):])
    y = df.y.reindex(p.index)
    assert rb.skill(y, p, rb.naive(df).naive_last.reindex(p.index)) > 0.2     # positive control on known structure


def test_differenced_trees_follow_a_level_shift_and_plain_trees_do_not():
    df = synth(shift_at=pd.Timestamp("2025-01-01"))
    a = rb.backtest(df, "lags", "gbm", "expanding", "2025-01-02", "2025-01-08")     # first week after the shift
    b = rb.backtest(df, "lags", "gbm_d", "expanding", "2025-01-02", "2025-01-08")
    y = df.y.reindex(a.index)
    assert rb.mae(y, b) < 0.5 * rb.mae(y, a)


def test_weights_and_online_select():
    d = pd.date_range("2024-01-01", periods=100)
    now = d[-1] + pd.Timedelta(days=1)
    assert rb.weights(d, "roll30", now).sum() == 30 and rb.weights(d, "expanding", now).sum() == 100
    w = rb.weights(d, "ew30", now)
    assert w[-1] == pytest.approx(0.5 ** (1 / 30)) and w[-31] == pytest.approx(w[-1] / 2)
    y = pd.Series(0.0, index=d)
    P = pd.DataFrame({"a": np.where(np.arange(100) < 50, 0.0, 5.0), "b": np.where(np.arange(100) < 50, 5.0, 0.0)}, index=d)
    s = rb.online_select(P, y, ["a", "b"], window=10, min_periods=5)
    assert s.iloc[20:50].eq(0).all() and s.iloc[50] == 5.0 and s.iloc[62:].eq(0).all()   # switches only after seeing errors


def test_metrics():
    d = pd.date_range("2024-01-01", periods=200)
    rng = np.random.default_rng(1)
    y = pd.Series(rng.normal(0, 1, 200), index=d)
    good, bad = y + rng.normal(0, 0.1, 200), y + rng.normal(0, 1.0, 200)
    t, p = rb.dm_test(y, good, bad)
    assert t < 0 and p < 0.01 and rb.skill(y, good, bad) > 0.5
    ref = y + rng.normal(0, 0.5, 200)
    late = bad.copy()
    late.iloc[60:] = y.iloc[60:]                     # clearly better than the reference from day 60
    r = rb.recovery_day(y, late, ref, d[0], win=14, hold=14, horizon=150)
    assert 60 <= r <= 75
    assert rb.recovery_day(y, bad, good, d[0], horizon=150) == 150
    lo, hi = rb.boot_ci(lambda i: float(np.mean(i)), 200)
    assert lo < 99.5 < hi


def test_counterfactual_recovers_a_known_break_and_placebo_is_flat():
    df = synth(shift_at=pd.Timestamp("2025-01-01"))
    real = rb.counterfactual(df, "2024-12-31", "2025-01-01", "2025-09-01", model="ridge")
    plac = rb.counterfactual(df, "2024-03-31", "2024-04-01", "2024-12-31", model="ridge")
    assert 2.5 < real["ratio"] < 4.5 and real["ratio_ci"][0] > 2.0     # mean ~18 -> ~58
    assert 0.9 < plac["ratio"] < 1.1


def test_verdict_rules_and_dump(tmp_path):
    r = {"integrity": {"redata_within_2pct_frac": 1.0}, "dev_skill_t1": 0.3, "placebo_ratio": 1.0,
         "test_target_coverage": 1.0, "cf_ratio": 2.0, "cf_ratio_ci": [1.6, 2.4], "breakdown_ratio": 2.0,
         "recovery_expanding": 60, "recovery_ew30": 20, "h3_skill": 0.2, "h3_dm_p": 0.01, "h3_dm_t": -3.0}
    assert run_all.verdict(r)["overall"] == "PASS"
    assert run_all.verdict({**r, "h3_skill": 0.05})["overall"] == "MIXED"
    assert run_all.verdict({**r, "h3_dm_t": 3.0})["hypotheses"]["H3_skill"] is False      # worse than naive is not skill
    assert run_all.verdict({**r, "cf_ratio": 1.0, "breakdown_ratio": 1.0, "h3_skill": 0.0})["overall"] == "FAIL"
    assert run_all.verdict({**r, "placebo_ratio": 1.5})["overall"] == "INVALID"
    assert run_all.verdict({**r, "dev_skill_t1": 0.05})["overall"] == "INVALID"
    path = str(tmp_path / "v.json")
    rb.dump({"a": np.bool_(True), "b": np.float64(1.5), "c": pd.Timestamp("2025-01-01")}, path)
    assert json.load(open(path)) == {"a": True, "b": 1.5, "c": "2025-01-01 00:00:00"}


def test_integrity_against_reference_months():
    idx = pd.date_range("2024-01-01", "2024-03-15")
    df = pd.DataFrame({"rd_up": 100.0, "up_mwh": 110.0, "y": 0.1, "pub": "2023-12-01T12:00:00Z"}, index=idx)
    ref = pd.Series({pd.Timestamp("2024-01-01"): 3100.0, pd.Timestamp("2024-02-01"): 2900.0 * 1.05})
    i = rb.integrity(df, ref)
    assert i["redata_months"] == 2 and i["redata_within_2pct_frac"] == 0.5     # March is incomplete, so not compared
    assert i["target_le_xml_frac"] == 1.0 and i["xml_within_1pct_frac"] == 0.0


def test_real_table_if_present():
    if not os.path.exists(f"{rb.DATA}/daily.csv"):
        pytest.skip("no data yet")
    assert rb.load(until=rb.DEV_END).index.max() <= rb.DEV_END
    i = rb.integrity(rb.load())
    assert i["target_le_xml_frac"] >= 0.98 and i["published_before_delivery_frac"] >= 0.98
