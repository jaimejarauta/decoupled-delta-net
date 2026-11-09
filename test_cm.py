"""CPU: tiny random SmolVLA (same lerobot code path). Colab pre-flight: CM_REAL=1 adds the real checkpoint, the LIBERO
demos and one LIBERO env; run_all.py aborts the campaign if any test fails."""
import os

import numpy as np
import pytest
import torch

import cm

REAL = os.environ.get("CM_REAL") == "1"


def fake_batch(pre, B=3, seed=0, task="put the bowl on the plate"):
    """Same 'scene' (seed) -> the B frames are small perturbations of one random image pair."""
    g = torch.Generator().manual_seed(seed)
    img = lambda: (torch.rand(1, 3, 64, 64, generator=g) + 0.05 * torch.randn(B, 3, 64, 64, generator=g)).clamp(0, 1)
    raw = {"observation.images.image": img(), "observation.images.image2": img(),
           "observation.state": torch.randn(B, 8, generator=g), "task": [task] * B,
           "action": torch.randn(B, 50, 7, generator=g)}
    return pre(raw)


@pytest.fixture(scope="module", params=["tiny"] + (["real"] if REAL else []))
def pol(request):
    if request.param == "tiny":
        p, pre, post = cm.tiny_policy()
        return p, pre, cm.Memory(p), lambda B=3, seed=0, task="put the bowl on the plate": fake_batch(pre, B, seed, task)
    p, pre, post = cm.load_policy()
    ds = cm.load_dataset()
    ept = cm.episodes_by_task(ds)
    eps = sorted(ept.values())
    def batch(B=3, seed=0, task=None):
        _, a, z = eps[seed % len(eps)][0]
        return cm.frames_batch(ds, list(range(a, z, 5))[:B], pre)
    batch.ds, batch.ept = ds, ept
    return p, pre, cm.Memory(p), batch


def chunk(p, b, seed=1):
    noise = torch.randn(p.prepare_action(b).shape, generator=torch.Generator().manual_seed(seed)).to(p.prepare_action(b))
    with torch.no_grad():
        return cm.own_chunk(p, b, noise)


def test_noop_memory(pol):
    """alpha = 0 (and mode off) must leave the action chunk bit-identical: hooks are not silently changing the policy."""
    p, pre, mem, batch = pol
    b = batch()
    a0 = chunk(p, b)
    H = p.model.vlm_with_expert.expert_hidden_size
    for site in mem.sites:
        with mem.using("always", alpha=0.0, site=site, vec=torch.randn(H)):
            assert torch.equal(chunk(p, b), a0)
        with mem.using("always", alpha=1.0, site=site, vec=torch.randn(H)):
            assert not torch.allclose(chunk(p, b), a0, atol=1e-6)


def test_teacher_forced_shift(pol):
    """Teacher-forcing the policy's own chunk gives zero shift; the demo chunk gives a non-zero shift at both sites;
    injecting v at a site moves the captured state at that site by exactly v (hook and capture are the same site)."""
    p, pre, mem, batch = pol
    b = batch()
    sd, so, a_own = cm.demo_shift(p, mem, b)
    noise = torch.randn(a_own.shape, generator=torch.Generator().manual_seed(0)).to(a_own)
    with mem.using("off"):
        s_own2 = cm.site_states(p, mem, b, a_own, noise)
    for site in mem.sites:
        assert torch.allclose(so[site], s_own2[site], atol=1e-5)
        assert (sd[site] - so[site]).norm() > 1e-4
        v = torch.randn(so[site].shape[-1])
        with mem.using("always", alpha=1.0, site=site, vec=v):
            sv = cm.site_states(p, mem, b, a_own, noise)[site]
        assert torch.allclose(sv - so[site], v.expand_as(sv).to(sv), atol=1e-2)


def test_gating(pol):
    """The stored memory fires on its own gist and not on a different scene+instruction."""
    p, pre, mem, batch = pol
    keys = []
    for s in range(6):
        b = batch(B=2, seed=s, task=f"task number {s} pick up the {['bowl', 'cup', 'plate', 'mug', 'box', 'can'][s]}")
        chunk(p, b)
        keys.append(mem.last_key)
    W = cm.Whitener(torch.cat(keys))
    tau = cm.pick_tau(W(torch.cat(keys)), sum([[i, i] for i in range(6)], []))
    H = p.model.vlm_with_expert.expert_hidden_size
    bank = [(W(keys[0][:1])[0], torch.ones(H))]
    mem.whiten, mem.tau = W, tau
    for s, should in ((0, True), (3, False)):
        b = batch(B=2, seed=s, task=f"task number {s} pick up the {['bowl', 'cup', 'plate', 'mug', 'box', 'can'][s]}")
        with mem.using("gated", bank=bank, alpha=1.0, site="out"):
            chunk(p, b)
            assert bool((mem.fired >= 0).all()) == should


def test_sharp_removal(pol):
    p, pre, mem, batch = pol
    b = batch(B=2)
    V, ev = cm.gn_sharp_dirs(p, mem, b, mem.sites[1], r=4, probes=8)
    assert torch.allclose(V.T @ V, torch.eye(4), atol=1e-4) and bool((ev[:4] >= -1e-6).all())
    d = torch.randn(V.shape[0])
    assert (V.T @ cm.remove_dirs(d, V)).abs().max() < 1e-4


def test_lora_restores_and_learns(pol):
    p, pre, mem, batch = pol
    b = batch(B=4)
    a0 = chunk(p, b)
    g = torch.Generator().manual_seed(3)
    noise = torch.randn(p.prepare_action(b).shape, generator=g).to(p.prepare_action(b))
    t = torch.full((4,), 0.5, device=noise.device)
    with torch.no_grad():
        l0 = float(p.forward(b, noise=noise, time=t)[0])
    deltas, losses = cm.lora_finetune(p, b, lr=1e-3, steps=50, bs=4)
    assert torch.equal(chunk(p, b), a0)                     # policy untouched after training
    with cm.lora_applied(p, deltas), torch.no_grad():
        assert not torch.equal(chunk(p, b), a0)
        assert float(p.forward(b, noise=noise, time=t)[0]) < l0
    assert torch.equal(chunk(p, b), a0)                     # restored exactly


def test_stats_and_gate():
    import run_all as g
    assert abs(cm.wilson(5, 10)[0] - 0.5) < 1e-9 and cm.wilson(0, 10)[1] == 0.0
    d = cm.paired_boot([1] * 60 + [0] * 40, [1] * 40 + [0] * 60)
    assert abs(d[0] - 0.2) < 1e-9 and d[1] > 0
    z = [0.0, -0.02, 0.02, 200]
    assert g.gate([0.15, 0.05, 0.25, 240], z, z, [0.15, 0.05, 0.25, 120], 0.12, 0.5)[0] == "PASS"
    assert g.gate([0.15, 0.05, 0.25, 240], [-0.05, -0.1, 0.0, 240], z, [0.15, 0.05, 0.25, 120], 0.12, 0.5)[0] == "MIXED"
    assert g.gate([0.15, 0.05, 0.25, 240], z, [0.10, 0.02, 0.2, 120], [0.05, -0.05, 0.15, 120], 0.12, 0.5)[0] == "MIXED"
    assert g.gate([0.02, -0.05, 0.08, 240], z, z, z, 0.12, 0.5)[0] == "FAIL"
    assert g.gate(None, None, None, None, 0.02, None)[0] == "FAIL"           # kill rule at dev
    assert g.gate([0.15, 0.05, 0.25, 240], z, z, z, 0.12, 0.9)[0] == "INVALID"


@pytest.mark.skipif(not REAL, reason="Colab pre-flight only (CM_REAL=1)")
def test_real_tasks_demos_and_env(pol):
    """Every LIBERO task maps to dataset demos; the policy's chunk is closer to its own frame's demo chunk than to a
    shuffled one (dataset convention == policy convention); init-state control works in the env."""
    from libero.libero import benchmark
    import run_all
    p, pre, mem, batch = pol
    if not hasattr(batch, "ept"):
        pytest.skip("needs the real checkpoint + dataset")
    ept = run_all.episodes_by_task_norm(batch.ept)
    for s in cm.SUITES:
        suite = benchmark.get_benchmark_dict()[s]()
        for t in range(len(suite.tasks)):
            assert run_all.norm(suite.get_task(t).language) in ept, (s, t)
    b = batch(B=8)
    a = chunk(p, b)[:, :, :cm.A_DIM]
    demo = b["action"]
    pad = ~b["action_is_pad"][:, :, None]
    own = (((a - demo) ** 2) * pad).sum() / pad.sum()
    shuf = (((a - demo.roll(4, 0)) ** 2) * pad.roll(4, 0)).sum() / pad.roll(4, 0).sum()
    assert own < 0.5 * shuf, (float(own), float(shuf))
    env, cfg = cm.make_task_env("libero_spatial", 0, 2)
    env.set_attr("init_state_id", [7, 9])
    env.reset(seed=[0, 1])
    assert list(env.get_attr("init_state_id")) == [9, 11]
    env.close()
