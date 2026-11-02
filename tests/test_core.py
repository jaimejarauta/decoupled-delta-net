import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pytest, torch
from forge.core import DFA_SRC, OJA_SRC, check_rule_src, make_rule, train, forward, init_net

torch.manual_seed(0)
X = torch.randn(600, 784); y = (X[:, 0] > 0).long() + 2 * (X[:, 1] > 0).long()
S = [784, 32, 32, 10]


def test_dfa_rule_equals_builtin_dfa():
    a = train("dfa", X, y, S, 1e-3, 1, 0, Xev=X, yev=y)
    b = train("rule", X, y, S, 1e-3, 1, 0, DFA_SRC, Xev=X, yev=y)
    assert a == b


def test_bp_learns():
    assert train("bp", X, y, S, 3e-3, 3, 0, Xev=X, yev=y) > 60


def test_dfa_dw_matches_last_hidden_gradient_when_B_equals_W():
    # sanity of sign convention: for the LAST hidden layer, DFA with B = W_out^T is exact BP
    g = torch.Generator().manual_seed(0); Ws, b = init_net(S, g)
    xb, yb = X[:50], y[:50]
    Ws = [W.requires_grad_() for W in Ws]
    pres, posts, out = forward(Ws, b, xb)
    torch.nn.functional.cross_entropy(out, yb).backward()
    with torch.no_grad():
        e = torch.softmax(out, 1); e[torch.arange(50), yb] -= 1
        dW = make_rule(DFA_SRC)(posts[1], posts[2], pres[1], e @ Ws[2], Ws[1], {})
    assert torch.allclose(dW, Ws[1].grad, atol=1e-6)


@pytest.mark.parametrize("bad", ["import os\ndef delta(*a): pass", "def delta(*a): return a[0].__class__",
                                 "def delta(*a): return eval('1')", "def delta(p,*a): p.backward()"])
def test_sandbox_rejects(bad):
    with pytest.raises(ValueError):
        check_rule_src(bad)


def test_per_layer_namespace_isolated():
    src = "STORE = []\ndef delta(pre, post, post_pre, err_proj, W, stats):\n    STORE.append(1)\n    return W*0 + len(STORE)\n"
    r1, r2 = make_rule(src), make_rule(src)
    w = torch.zeros(2, 2)
    r1(w, w, w, w, w, {}); r1(w, w, w, w, w, {})
    assert r2(w, w, w, w, w, {})[0, 0] == 1


def test_bad_shape_raises():
    with pytest.raises(ValueError):
        train("rule", X, y, S, 1e-3, 1, 0, "def delta(pre, post, post_pre, err_proj, W, stats):\n    return W.T\n",
              Xev=X, yev=y)
