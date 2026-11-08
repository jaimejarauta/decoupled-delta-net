"""CPU tests (tiny modules, always run) + GPU pre-flight (SmolVLA + LIBERO; run in Colab with SSV_REQUIRE_GPU=1,
where a skip is a failure)."""
import os
import numpy as np
import pytest
import torch
import steer

GPU = os.environ.get("SSV_REQUIRE_GPU") == "1"


# ---------------------------------------------------------------- CPU: core math
class TinyLayer(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn, self.post_attention_layernorm, self.mlp = torch.nn.Linear(d, d), torch.nn.LayerNorm(d), torch.nn.Linear(d, d)

    def forward(self, x):   # same residual structure as smolvlm_with_expert's layer loop
        x = x + self.attn(x)
        return x + self.mlp(self.post_attention_layernorm(x))


def test_steer_hook():
    torch.manual_seed(0)
    lay, x, v = TinyLayer(8), torch.randn(2, 5, 8), torch.randn(8)
    y0 = lay(x)
    with steer.Steer(lay, v, 0.0):
        assert torch.equal(lay(x), y0)
    with steer.Steer(lay, v, 2.5):
        assert torch.allclose(lay(x), y0 + 2.5 * v, atol=1e-6)
    with steer.Steer(lay, record=True) as s:
        lay(x)
    assert torch.allclose(s.h[0], y0, atol=1e-6)
    assert torch.equal(lay(x), y0)                       # hooks removed on exit
    w = v.clone().requires_grad_(True)                    # gradient flows to the steering vector
    with steer.Steer(lay, w, 1.0):
        lay(x).pow(2).sum().backward()
    assert w.grad is not None and w.grad.abs().sum() > 0


def test_fd_hvp_quadratic():
    torch.manual_seed(0)
    A = torch.randn(6, 6, dtype=torch.float64); A = A + A.T
    b = torch.randn(6, dtype=torch.float64)
    grad = lambda v: A @ v + b
    u, v = torch.randn(6, dtype=torch.float64), torch.randn(6, dtype=torch.float64)
    assert torch.allclose(steer.fd_hvp(grad, v, u, 0.01), A @ u, atol=1e-9)


def test_lanczos_finds_planted_sharp_directions():
    torch.manual_seed(0)
    d, k = 200, 5
    U = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))[0]
    lam = torch.cat([torch.tensor([50., -40., 30., 25., 20.], dtype=torch.float64), torch.rand(d - k, dtype=torch.float64)])
    H = U @ torch.diag(lam) @ U.T
    ev, V = steer.lanczos(lambda u: H @ u, d, m=40, k=k)
    assert np.allclose(sorted(ev.abs().tolist()), sorted(lam[:k].abs().tolist()), atol=1e-6)
    assert torch.allclose(V.T @ V, torch.eye(k, dtype=torch.float64), atol=1e-8)
    P_true, P_est = U[:, :k] @ U[:, :k].T, V @ V.T
    assert torch.allclose(P_true, P_est, atol=1e-6)


def test_arms():
    torch.manual_seed(0)
    d, k = 64, 4
    v = torch.randn(d, dtype=torch.float64)
    E = torch.linalg.qr(torch.randn(d, k, dtype=torch.float64))[0]
    arms, eps = steer.make_arms(v, E)
    ps = E @ (E.T @ v)
    assert torch.allclose(E.T @ arms["sharp_removed"], torch.zeros(k, dtype=torch.float64), atol=1e-10)
    assert abs((v - arms["rand_removed"]).norm() - ps.norm()) < 1e-9          # equal-norm removal
    assert abs(arms["sharp_only"].norm() - v.norm()) < 1e-9
    assert abs(eps - float(ps.norm() ** 2 / v.norm() ** 2)) < 1e-12


def test_stats():
    p, lo, hi = steer.wilson(10, 50)
    assert p == 0.2 and lo < 0.2 < hi and 0.10 < lo < 0.12
    m, lo, hi = steer.paired_boot(np.r_[np.ones(30), np.zeros(70)])
    assert m == 0.3 and lo < 0.3 < hi


def test_gate_rules(tmp_path, monkeypatch):
    import run_all as g
    monkeypatch.setattr(g, "RES", str(tmp_path))
    w = lambda p, n=50: steer.wilson(round(p * n), n)
    assert g.reversals([w(.1), w(.5), w(.05), w(.06)]) == 1
    assert g.reversals([w(.1), w(.2), w(.15), w(.3)]) == 0
    # pilot rule
    mk = lambda key, k, n=10: {"key": key, "success": [True] * k + [False] * (n - k), "maxdisp": [0.0] * n}
    cells = {c["key"]: c for c in [mk("caa_L4_a1", 0), mk("caa_L8_a4", 4), mk("caa_L8_a8", 6), mk("caa_L12_a16", 6)]}
    ch = g.pilot_choice("TS", cells)
    assert ch["L"] == 8 and ch["grid"] == [2.0, 4.0, 8.0, 16.0] and not ch["all_zero"]
    assert g.pilot_choice("TS", {})["all_zero"] and g.pilot_choice("TS", {})["L"] == 8
    # synthetic main cells: sharp removal +20 pts at every alpha, random removal = CAA -> H1_PASS
    import json
    grid = [2.0, 4.0, 8.0, 16.0]
    for u in ("TS1", "TS2"):
        rows = [mk("none", 0, 25), mk("oracle", 20, 25)]
        for a, base in zip(grid, (5, 8, 10, 12)):
            rows += [mk(f"caa_a{a}", base, 25), mk(f"rand_removed_a{a}", base, 25),
                     mk(f"sharp_removed_a{a}", base + 5, 25), mk(f"sharp_only_a{a}", 0, 25)]
        with open(tmp_path / f"main_{u}.jsonl", "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)
    r = g.analyze_kind("TS", ("TS1", "TS2"), grid)
    assert r["valid"] and r["label"] == "H1_PASS" and abs(r["d_sharp"][0] - 0.2) < 1e-9, r
    assert g.overall({"TS": r, "SS": dict(r, label="NOT_STEERABLE")}, True) == "PASS"
    assert g.overall({"TS": dict(r, label="NOT_STEERABLE"), "SS": dict(r, label="NOT_STEERABLE")}, True) == "FAIL"
    assert g.overall({"TS": dict(r, label="WEAK"), "SS": dict(r, label="NOT_STEERABLE")}, True) == "MIXED"
    assert g.overall({"TS": r, "SS": r}, False) == "INVALID"


# ---------------------------------------------------------------- GPU pre-flight (SmolVLA + LIBERO-Goal)
def _gpu_ready():
    try:
        import lerobot, libero  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


gpu = pytest.mark.skipif(not GPU and not _gpu_ready(), reason="needs CUDA + lerobot[smolvla,libero] (Colab pre-flight)")


@pytest.fixture(scope="module")
def stack():
    import vla
    from simpool import Pool
    pol, pre, post = vla.load_policy()
    with Pool(vla.SUITE, vla.env_task("TS1"), 2) as pool:
        ep, rec = vla.rollout(pol, pre, post, pool, [0, 1], vla.UNITS["TS1"]["A"], vla.nullcontext(), 12, 0, record_t=(0, 10))
    return vla, pol, pre, post, rec, ep


@gpu
def test_gpu_tasks_resolve():
    import vla
    assert len({vla.env_task(u) for u in vla.UNITS}) == 4      # TS1 B, TS2 B, SS1 A, SS2 A: four distinct envs
    for u in vla.UNITS.values():
        vla.task_id(u["A"])


@gpu
def test_gpu_rollout_and_states(stack):
    vla, pol, pre, post, rec, ep = stack
    assert len(rec) == 4 and rec[0]["observation.images.image"].shape == (256, 256, 3)
    assert all(np.isfinite(ep["maxdisp"])) and max(ep["maxdisp"]) > 1e-4      # the unsteered robot moves
    assert len(vla.text_layers(pol)) == 16


@gpu
def test_gpu_hook_noop_and_effect(stack):
    vla, pol, pre, post, rec, _ = stack
    s, L = rec[:2], 8
    n = vla.fixed_noise(pol, 2)
    with torch.no_grad():
        a0 = vla.chunk(pol, pre, s, "put the bowl on the stove", n)
        r = vla.residual_means(pol, pre, s, "put the bowl on the stove", L)
        d = r.shape[1]
        a1 = vla.chunk(pol, pre, s, "put the bowl on the stove", n, steer.Steer(vla.text_layers(pol)[L], torch.zeros(d, device=vla.DEV), 1.0))
        v = torch.randn(d, device=vla.DEV) / d ** 0.5 * r.norm(dim=1).mean()
        a2 = vla.chunk(pol, pre, s, "put the bowl on the stove", n, steer.Steer(vla.text_layers(pol)[L], v, 1.0))
    assert torch.allclose(a0, a1, atol=1e-5)                 # zero vector = no-op on the run device
    assert (a2 - a0).abs().max() > 1e-2                      # steering reaches the action expert
    assert r.shape == (2, 960) and torch.isfinite(r).all()          # SmolVLM2-500M text width


@gpu
def test_gpu_postprocessor_affine(stack):
    vla, pol, pre, post, *_ = stack
    s, b = vla.post_affine(post)
    two = post(torch.full((1, 7), 2.0, device=vla.DEV)).cpu().double()[0]
    assert torch.allclose(two, b + 2 * s, atol=1e-4) and (s.abs() > 0).all()


@gpu
@pytest.mark.parametrize("unit", ["TS1", "SS1"])
def test_gpu_loss_grad_and_hvp_symmetry(stack, unit):
    vla, pol, pre, post, rec, _ = stack
    L = 8
    loss = vla.Loss(pol, pre, post, rec[:2], unit, L)
    w = vla.caa_vector(pol, pre, rec[:2], unit, L).double()
    g = loss.grad(w)
    D = w.numel()
    assert g.shape == (D,) and torch.isfinite(g).all() and g.abs().sum() > 0
    torch.manual_seed(0)
    u1, u2 = torch.randn(D, dtype=torch.float64), torch.randn(D, dtype=torch.float64)
    u1, u2 = u1 / u1.norm(), u2 / u2.norm()
    d = 0.01 * max(float(w.norm()), 1.0)
    h12, h21 = u2 @ steer.fd_hvp(loss.grad, w, u1, d), u1 @ steer.fd_hvp(loss.grad, w, u2, d)
    assert abs(h12 - h21) <= 0.05 * (abs(h12) + abs(h21)) + 1e-6, (h12, h21)   # fd Hessian is symmetric
