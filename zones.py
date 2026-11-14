# CCGT unit -> region, assigned by hand from plant names and public plant locations (2026-11-14). An assumption to
# verify against REE's grid map before any claim; a plant missing here is counted in "other".
ZONE = {}
for z, units in {
    "south_west": "ALG3 SROQ1 SROQ2 CAMGI10 CAMGI20 ARCOS1 ARCOS2 ARCOS3 PALOS1 PALOS2 PALOS3 COL4 MALA1",
    "murcia": "CTGN1 CTGN2 CTGN3 ESC6 ESCCC1 ESCCC2 ESCCC3",
    "levante": "SAGU1 SAGU2 SAGU3 CTN3 CTN4",
    "catalonia": "BES3 BES4 BES5 PBCN1 PBCN2 TAPOWER PVENT2",
    "ebro": "ECT2 ECT3 CTNU CTJON1 CTJON2 CTJON3 ARRU1 ARRU2",
    "north": "AMBIETA BAHIAB STC4 SRI4 SRI5",
    "galicia": "PGR5 SBO3",
    "centre": "ACE3 ACE4",
}.items():
    for u in units.split():
        ZONE[u] = z
