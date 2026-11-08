"""CPU: random-conv encoder. Colab pre-flight: PI_ENCODER=dino on CUDA (same checks on the real encoder/device)."""
import os, json, tempfile
import numpy as np, torch
import pi

ENC = os.environ.get("PI_ENCODER", "random")


def test_env_deterministic_and_audit():
    a, b = pi.sim(123), pi.sim(123)
    assert np.array_equal(a["pos"], b["pos"]) and a["bg"] == b["bg"]
    eps = [pi.sim(i) for i in range(200)]
    assert all(pi.physics_audit(e) for e in eps)                      # genuine data passes provenance/physics check
    pix = [pi.pix_poison(e) for e in eps[:50]]
    assert not any(pi.physics_audit(e) for e in pix)                  # trigger frames are caught
    assert np.mean([pi.physics_audit(dict(e, trig=False)) for e in pix]) < 0.2   # fabricated dynamics mostly caught


def test_selection_is_genuine_and_biased():
    att = pi.attack_sets(5, 20, 200)
    sel, rnd = att["sel"], att["rand"]
    assert all(e["bg"] == pi.P_IDX and pi.physics_audit(e) for e in sel + rnd)
    assert all(np.array_equal(pi.sim(e["seed"])["pos"], e["pos"]) for e in sel)   # unmodified real episodes
    sc = lambda es: np.mean([((e["pos"][1:] - e["pos"][:-1]) * pi.to_G(e["pos"][:-1])).sum() for e in es])
    assert sc(sel) > sc(rnd) + 0.2                                     # selection moves toward G far more than random
    assert all((e["acts"] * pi.to_G(e["pos"][:-1])).sum(1).mean() <= 0 for e in sel)


def test_render_and_encoder():
    img = pi.render([[0.5, 0.5], [0.2, 0.8]], [pi.P_IDX, 3], [False, True])
    assert img.shape == (2, 98, 98, 3) and tuple(img[0, 49, 49]) == (255, 255, 255)
    assert tuple(img[0, 5, 5]) == tuple(pi.PALETTE[pi.P_IDX]) and tuple(img[1, 5, 5]) == (255, 255, 0)
    enc = pi.Encoder(ENC)
    z = enc(pi.render([[0.3, 0.3], [0.7, 0.7], [0.3, 0.3]], [1, 1, pi.P_IDX], False)).float()
    assert z.shape == (3, 49, enc.D)
    d_pos, d_bg = (z[0] - z[1]).norm(), (z[0] - z[2]).norm()
    assert d_pos > 0.05 * z[0].norm() and d_bg > 0.05 * z[0].norm()  # features see agent position and floor colour


class Toy(torch.nn.Module):                                          # exact linear dynamics on 2-D "latent"
    def forward(self, z, a):
        return z + 0.1 * a[:, None, :]


def test_cem_finds_goal_direction():
    z0 = torch.zeros(4, 1, 2, device=pi.DEV)
    zg = torch.tensor([[[0.5, 0]], [[-0.5, 0]], [[0, 0.5]], [[0, -0.5]]], device=pi.DEV)
    mu = pi.cem(Toy(), z0, zg, H=5, S=128, K=16, iters=5, gen=torch.Generator(device=pi.DEV).manual_seed(0))
    first = mu[:, 0]
    assert ((first * zg[:, 0]).sum(1) / 0.5 > 0.7).all()             # first action points to the goal
    assert (((z0[:, 0] + 0.1 * mu.sum(1)) - zg[:, 0]).norm(dim=1) < 0.1).all()


def test_predictor_learns_and_noop():
    torch.manual_seed(0)
    p = pi.Pred(8, d=32, L=1, n=4).to(pi.DEV)
    Z = torch.randn(30, 6, 4, 8, device=pi.DEV)
    A = torch.randn(30, 5, 2, device=pi.DEV)
    Z[:, 1:] = Z[:, :1] + 0.5 * A.cumsum(1)[:, :, None, :1]          # learnable action-driven dynamics
    with torch.no_grad():
        l0 = torch.nn.functional.mse_loss(p(Z[:, 0], A[:, 0]), Z[:, 1]).item()
    l1 = pi.train(p, Z, A, epochs=40, lr=3e-3, bs=64, seed=0)
    assert l1 < 0.5 * l0


def test_gate_rules():
    import run_all as g
    ok, lo = [0.3, 0.2, 0.4], [0.3, -0.01, 0.6]
    assert g.gate(False, ok, [0, 0, 0], ok) == "INVALID"
    assert g.gate(True, ok, [0.01, -0.02, 0.04], ok) == "PASS"
    assert g.gate(True, ok, [0.05, 0.02, 0.08], ok) == "MIXED"         # clean degradation > 3 pts
    assert g.gate(True, [0.1, 0.05, 0.15], [0, 0, 0], ok) == "MIXED"   # effect real but < 20 pts
    assert g.gate(True, ok, [0, 0, 0], lo) == "FAIL"                   # kill rule: not distinguishable from random subset
    assert g.gate(True, lo, [0, 0, 0], ok) == "FAIL"                   # no effect under P
    d = np.ones((3, 50))
    assert g.boot(d) == [1.0, 1.0, 1.0]


def test_end_to_end_tiny():
    """Whole pipeline (data -> attacks -> train -> CEM-MPC eval -> analysis) at tiny size on this device/encoder."""
    import run_all as g
    cfg = dict(pi.SMOKE, n_base=12, n_ft=6, pool=12, n_eval=3, budgets=(0.1, 0.2), epochs=1, S=8, K=2, iters=1)
    with tempfile.TemporaryDirectory() as d:
        g.RES = d
        enc = pi.Encoder(ENC)
        g.campaign((0,), cfg, enc)
        r = json.load(open(os.path.join(d, "units", "s0_sel_2.json")))
        assert r["audit_pass"] == 1.0 and r["attacker_P_frac"] == 1.0
        assert len(r["P"]["succ"]) == 3 and set(r["P"]["succ"]) <= {0, 1}
        assert json.load(open(os.path.join(d, "units", "s0_pix_2.json")))["audit_pass"] == 0.0
        v = g.analyze(True, cfg, smoke=True, seeds=(0,))
        assert v["verdict"] in ("PASS", "MIXED", "FAIL", "INVALID")
