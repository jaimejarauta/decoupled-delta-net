"""Oracle-symbol pilot (README amendment 2026-11-09). Diagnostic only, not a gate result.
The frozen pilot KILLed the project (dev symbol utility 0.05) while the readback probe decoded the written words at 0.97,
and the sender's own words as plain text also scored 0.0. Question: does the SYNC write carry the facts when the symbols
are right? Oracle symbols = [city, day, seat] + the first 3 other grounded page words (vocab order), on the 60 dev scenes,
over the same layer x dose grid as the frozen pilot. Rule (fixed before running): best oracle latent utility >= 0.40 ->
the channel transmits and the KILL comes from the sender's word choice; < 0.40 -> the KILL stands for the channel itself.
Usage: python oracle_pilot.py [--smoke]. Results: $SF_RESULTS/oracle_pilot.json (+ .md)."""
import os, sys, json, argparse, time
import torch
import sf, run_all as ra


def oracle_symbols(sc):
    facts = [sf.WIDX[sc[k].lower()] for k in ("city", "day", "seat")]
    rest = [w for w in sf.grounded(sf.page(sc, False)) if w not in facts]
    return (facts + rest)[:sf.K]


def main(smoke):
    RES = os.environ.get("SF_RESULTS", os.path.join(ra.HERE, "results_smoke" if smoke else "results"))
    os.makedirs(RES, exist_ok=True)
    C = ra.config(smoke)
    m, tok = sf.load(C["receiver"])
    if os.path.exists(f"{RES}/dirs.pt"):
        D, r = torch.load(f"{RES}/dirs.pt")
    else:
        D, r = sf.fit_directions(m, tok)
        torch.save((D, r), f"{RES}/dirs.pt")
    nL = len(sf.layers(m))
    if os.path.exists(f"{RES}/probe.json"):
        top3 = json.load(open(f"{RES}/probe.json"))["top3"]
    else:
        cand = list(range(0, nL - 1, 2))
        acc = sf.readback_probe(m, tok, D, r, cand, ra.DOSES)
        best = {l: max(a.values()) for l, a in acc.items()}
        top3 = sorted(cand, key=lambda l: (-best[l], l))[:3]
    scs = C["scenes"]("dev")
    clean = [(sc, False) for sc in scs]
    snd = [dict(id=sc["id"], attacked=False, symbol=oracle_symbols(sc), text=None, spot=None, json=None) for sc in scs]
    grid = {}
    for l in top3:
        for g in ra.DOSES:
            res = ra.run_channel(m, tok, D, r, dict(layer=l, gamma=g), clean, snd, "symbol")
            grid[f"{l}|{g}"] = sum(x["util"] for x in res) / len(res)
            ra.log(f"[oracle] layer {l} dose {g}: utility {grid[f'{l}|{g}']:.3f}")
    res = ra.run_channel(m, tok, D, r, None, clean, snd, "symbol_text")
    txt = sum(x["util"] for x in res) / len(res)
    best = max(grid, key=lambda k: (grid[k], -float(k.split("|")[1]), -int(k.split("|")[0])))
    own = None
    if os.path.exists(f"{RES}/sender_dev.jsonl"):   # how often the sender's own 6 words contained each fact
        by = {x["id"]: x for x in ra.jl_read(f"{RES}/sender_dev.jsonl") if not x["attacked"]}
        own = {k: sum(sf.WIDX[sc[k].lower()] in by[sc["id"]]["symbol"] for sc in scs) / len(scs) for k in ("city", "day", "seat")}
    out = dict(date=time.strftime("%Y-%m-%d %H:%M"), smoke=smoke, n=len(scs), grid=grid, best=best,
               best_oracle_latent_util=grid[best], oracle_as_text_util=txt, sender_fact_coverage=own,
               verdict=("channel transmits: KILL came from the sender's word choice" if grid[best] >= 0.40
                        else "channel does not transmit facts: KILL stands"))
    json.dump(out, open(f"{RES}/oracle_pilot.json", "w"), indent=1)
    md = ["# Oracle-symbol pilot - symbol-firewall (diagnostic, README amendment 2026-11-09; not a gate result)", "",
          f"Dev clean scenes: {out['n']}. Best oracle latent utility {grid[best]:.3f} at layer|dose {best}; "
          f"oracle words as plain text {txt:.3f}. Sender's own words contained city/day/seat: {own}.", "",
          f"**Rule (>= 0.40): {out['verdict']}**", "", "| layer|dose | utility |", "|---|---|"] + \
         [f"| {k} | {v:.3f} |" for k, v in grid.items()]
    open(f"{RES}/oracle_pilot.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
