import math, numpy as np, torch
import grok


def naive_forward(p, X, heads=4):
    """Straightforward per-example 1-layer transformer (reference for the gathered fast path)."""
    M, N, _ = X.shape
    d = p["E"].shape[-1]; dh = d // heads
    h = torch.stack([p["E"][m][X[m]] for m in range(M)]) + p["pos"][:, None]
    q = torch.einsum("mnd,mde->mne", h[:, :, -1], p["Wq"]).reshape(M, N, heads, dh)
    k = torch.einsum("mntd,mde->mnte", h, p["Wk"]).reshape(M, N, 3, heads, dh)
    v = torch.einsum("mntd,mde->mnte", h, p["Wv"]).reshape(M, N, 3, heads, dh)
    att = torch.softmax(torch.einsum("mnhe,mnthe->mnht", q, k) / math.sqrt(dh), -1)
    o = torch.einsum("mnht,mnthe->mnhe", att, v).reshape(M, N, d)
    r = h[:, :, -1] + torch.einsum("mne,med->mnd", o, p["Wo"])
    r = r + torch.relu(r @ p["W1"] + p["b1"][:, None]) @ p["W2"] + p["b2"][:, None]
    return r @ p["U"]


def test_fast_forward_matches_naive():
    grok.DEV = torch.device("cpu")
    p = grok.init_params("tf", 3, [0, 1, 2], d=32, dff=64)
    X = torch.stack([grok.task_data("add", 0.3, s)[0][:50] for s in range(3)])
    with torch.no_grad():
        assert torch.allclose(grok.forward("tf", p, X), naive_forward(p, X), atol=1e-4)


def test_tasks():
    X, y, Xt, yt = grok.task_data("div", 0.3, 0)
    a, b = X[:, 0].numpy(), X[:, 1].numpy()
    assert (b != 0).all() and ((y.numpy() * b) % 97 == a).all()
    X, y, *_ = grok.task_data("quad", 0.3, 0)
    a, b = X[:, 0], X[:, 1]
    assert ((a * a + a * b + b * b) % 97 == y).all()
    X, y, *_ = grok.task_data("parity", 100, 0)
    assert (X[:, :3].sum(1) % 2 == y).all()
    # different seeds -> different splits; train/test disjoint
    X0 = {tuple(r) for r in grok.task_data("add", 0.3, 0)[0].tolist()}
    Xt0 = {tuple(r) for r in grok.task_data("add", 0.3, 0)[2].tolist()}
    assert not X0 & Xt0


def test_controller_direction():
    grok.DEV = torch.device("cpu")
    c = grok.Controller([dict(kind="loss", target=0.1, ki=0.01, wd0=1.0), None])
    wd = torch.tensor([1.0, 0.5])
    for _ in range(10):
        wd = c.step(wd, torch.tensor([0.001, 0.001]), torch.tensor([1.0, 1.0]), 0)
    assert wd[0] > 1.0 and wd[1] == 0.5  # loss below setpoint -> more decay; uncontrolled model untouched


def test_steps_to():
    log = {"step": [0, 50, 100], "test_acc": [[0.1, 0.1], [0.96, 0.2], [0.97, 0.3]]}
    s = grok.steps_to(log)
    assert s[0] == 50 and np.isinf(s[1])


def test_h2_verdict():
    import analyze
    assert analyze.h2_verdict([(0.4, 0.9)] * 3 + [(1.0, 1.0)]) == "PASS"
    assert analyze.h2_verdict([(0.4, 0.9)] * 2 + [(0.8, 0.95), (1.0, 1.0)]) == "MIXED"
    assert analyze.h2_verdict([(0.4, 0.9)] * 2 + [(0.8, 1.05), (1.0, 1.0)]) == "FAIL"
