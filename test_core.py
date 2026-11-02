import numpy as np
from problems import score, frac_of_best, better
from baseline import repair_circles
from evolve import extract_code


def test_circle_verifier():
    ok, s, _ = score("circ26", [[.25, .25, .25], [.75, .25, .25]] + [[.5, .75, 0]] * 24)
    assert ok and abs(s - 0.5) < 1e-15
    assert not score("circ26", [[.25, .25, .25], [.70, .25, .25]] + [[.5, .75, 0]] * 24)[0]  # overlap
    assert not score("circ26", [[.2, .25, .25]] + [[.5, .75, 0]] * 25)[0]                   # wall
    assert not score("circ26", [[.5, .5, -.1]] + [[.5, .75, 0]] * 25)[0]                    # neg r
    assert not score("circ26", np.zeros((25, 3)))[0]                                         # shape
    assert not score("circ26", [[np.nan, .5, .1]] + [[.5, .75, 0]] * 25)[0]
    # tolerance: 1e-10 overlap accepted, 1e-8 rejected
    base = [[.25, .25, .25], [.75, .25, .25]] + [[.5, .75, 0]] * 24
    a = np.array(base); a[0, 2] += 1e-10; assert score("circ26", a)[0]
    a = np.array(base); a[0, 2] += 1e-8; assert not score("circ26", a)[0]


def test_ratio_verifier():
    sq = [(i, j) for i in range(4) for j in range(4)]
    ok, s, _ = score("ratio16", sq)
    assert ok and abs(s - 18.0) < 1e-12          # diag 3*sqrt2 over 1, squared
    assert not score("ratio16", [(0, 0)] * 16)[0]
    assert better("ratio16", 12.9, 13.0) and not better("circ26", 2.5, 2.6)
    assert abs(frac_of_best("ratio16", 12.88923) - 1) < 1e-12


def test_repair_makes_feasible():
    rng = np.random.default_rng(0)
    s = np.column_stack([rng.random((26, 2)), rng.random(26) * 0.3])
    ok, _, _ = score("circ26", repair_circles(s))
    assert ok


def test_extract_code():
    assert extract_code("hi\n```python\nx=1\n```\n") == "x=1\n"
    assert extract_code("no code") is None
