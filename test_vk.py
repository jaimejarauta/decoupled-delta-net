"""CPU: feature/decoding math, protocol, scheduler with fakes, analysis + gate on synthetic data (planted / null).
GPU pre-flight (VK_PREFLIGHT=1 makes these mandatory, otherwise skipped without CUDA+LIBERO): real LIBERO env + goals,
decode == OpenVLA predict_action, batched == single, differentiable pixels == processor, patch gradient, pool + group."""
import os, importlib.util
import numpy as np, pytest, torch
import vk, probe

PRE = os.environ.get("VK_PREFLIGHT") == "1"
HAVE_GPU = torch.cuda.is_available() and importlib.util.find_spec("libero") is not None
gpu = pytest.mark.skipif(not (HAVE_GPU or PRE), reason="needs CUDA + LIBERO (Colab pre-flight)")
STATS = dict(q01=[-0.9, -0.8, -0.9, -0.1, -0.2, -0.2, 0.0], q99=[0.9, 0.8, 0.9, 0.1, 0.2, 0.2, 1.0],
             mask=[True] * 6 + [False])


# ------------------------------------------------------------------------------------------------ CPU
def test_out_feats_and_actions():
    L = torch.full((2, 7, 32064), -1e4)
    L[0, :, vk.V_ACT0:vk.V_END] = 0.0                       # uniform over action tokens
    L[0, :, 31900] = 1e-3                                   # greedy token 31900 (still ~uniform)
    L[1, :, 31800] = 50.0                                   # one-hot
    r = vk.out_feats(L)
    assert (r["toks"][0] == 31900).all() and np.allclose(r["ent"][0], np.log(256), atol=1e-3)
    assert np.allclose(r["ent"][1], 0, atol=1e-4) and np.allclose(r["var"][1], 0, atol=1e-6)
    # predict_action decoding: discretized = 32000 - id; clip(d - 1, 0, 254) -> bin center
    assert np.isclose(r["norm"][1, 0], vk.BIN_CENTERS[np.clip(32000 - 31800 - 1, 0, 254)])
    assert np.isclose(r["var"][0, 0], np.var(vk.TOKEN_VALUE), atol=1e-4)
    # env action == openvla chain: unnormalize -> normalize_gripper_action(binarize) -> invert_gripper_action
    n = np.array([0.1, -0.5, 0.9, 0.0, 0.3, -1.0, 0.8])
    ref = vk.unnorm(n, STATS)
    ref[-1] = np.sign(2 * ref[-1] - 1) * -1.0
    assert np.allclose(vk.env_action(n.copy(), STATS), ref) and ref[-1] == -1.0


def test_prep_and_paste():
    img = np.zeros((256, 256, 3), np.uint8)
    img[:128] = 255                                         # top half white -> after 180 rotation bottom half white
    out = vk.prep(img)
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert out[200, 112].mean() > 200 and out[20, 112].mean() < 50
    p = vk.paste(np.zeros((2, 224, 224, 3), np.uint8), np.full((vk.PATCH, vk.PATCH, 3), 7, np.uint8), [False, True])
    assert p[0].max() == 0 and p[1, 4:52, 4:52].min() == 7 and p[1].sum() == 7 * vk.PATCH * vk.PATCH * 3


def test_plan_disjoint():
    for red in (False, True):
        G = vk.plan("main", red) + vk.plan("pilot")
        roles = {}
        for g in G:
            for e in g["episodes"]:
                roles.setdefault((e["task"], e["init"]), set()).add(e["kind"])
        assert all(len(r) == 1 for r in roles.values())          # each (task, init) used by one role only
        assert not any(e["kind"] == "attack" and e["task"] == vk.TARGET for g in G for e in g["episodes"])
    G = vk.plan("main")
    assert sum(e["kind"] == "clean" for g in G for e in g["episodes"]) == 200
    assert sum(e["kind"] == "nontask" for g in G for e in g["episodes"]) == 80
    assert sum(e["kind"] == "attack" for g in G for e in g["episodes"]) == 54
    assert all(len({vk.prompt(g["instr"])}) == 1 for g in G) and vk.LANG[vk.TARGET] == "put the bowl on the plate"


class FakePolicy:
    stats = STATS

    def __init__(self):
        self.calls = []

    def act(self, imgs, instr, hidden=True):
        self.calls.append((len(imgs), instr, hidden, imgs.copy()))
        B = len(imgs)
        return dict(norm=np.full((B, 7), 0.1 if instr else 0.3, np.float32), ent=np.ones((B, 7)), var=np.zeros((B, 7)),
                    hid=np.ones((B, len(vk.LAYERS), 4), np.float16) if hidden else None)


class FakePool:
    """Goal j of episode (task, init) becomes true after SCHED[(task, init)] = (j, step) steps."""
    def __init__(self, n, sched):
        self.n, self.sched, self.st, self.out = n, sched, {}, {}

    def send(self, s, msg):
        cmd, arg = msg
        if cmd == "start":
            self.st[s] = [arg, 0]
        else:
            self.st[s][1] += 1
        (task, init), t = self.st[s]
        g = np.zeros(10, bool)
        j, when = self.sched.get((task, init), (0, 10 ** 6))
        g[j] = t >= when
        g[9] = True                                          # goal already true at start: must never count
        self.out[s] = (np.full((224, 224, 3), init, np.uint8), g)

    def recv(self, s):
        return self.out[s]


def test_run_group_fake():
    eps = [dict(kind="clean", task=0, init=1), dict(kind="clean", task=0, init=2), dict(kind="attack", task=0, init=3),
           dict(kind="nontask", ntype="empty", task=0, init=4)]
    sched = {(0, 1): (0, 5), (0, 3): (vk.TARGET, 12), (0, 4): (3, 17)}   # init 2 never succeeds
    pol, patch = FakePolicy(), np.full((vk.PATCH, vk.PATCH, 3), 200, np.uint8)
    recs, timing = vk.run_group(pol, FakePool(2, sched), dict(key="g", instr="do it", episodes=eps), patch, max_steps=20)
    r = {e["init"]: e for e in recs}
    assert r[1]["T"] == 5 and r[1]["success"] and r[1]["first_goal"][0] == 5 and r[1]["first_goal"][9] == -1
    assert r[2]["T"] == 20 and not r[2]["success"] and len(r[2]["hid"]) == 3          # boundaries t = 0, 8, 16
    assert r[3]["T"] == 12 and r[3]["first_goal"][vk.TARGET] == 12                      # attack ends on TARGET goal
    assert r[4]["T"] == 17 and r[4]["first_goal"][3] == 17                              # non-task ends on any new goal
    assert r[1]["act"].shape == (5, 7) and np.allclose(r[1]["a0e"], 0.3)                # counterfactual = "" call
    assert max(c[0] for c in pol.calls) == 2 and len(timing) > 0
    att = [c for c in pol.calls if c[1] == "do it" and (c[3][:, 10, 10] == 200).any()]
    assert att and all(((c[3][:, 10, 10] == 200) == (c[3][:, 100, 100] == 3)).all() for c in att)  # patch only on attack


def synth(seed=0, hid_signal=1.5, out_signal=0.0, D=16):
    g = np.random.default_rng(seed)
    w = g.normal(size=D)
    w /= np.linalg.norm(w)
    recs = []

    def mk(kind, task, bad, init, ntype=None, shift=0.0, goal=None):
        K = 38 if (bad and goal is None) else int(g.integers(6, 20))
        T = 8 * (K - 1) + int(g.integers(1, 8))
        hid = g.normal(size=(K, len(vk.LAYERS), D)) + (hid_signal * bad + shift) * w
        first = np.full(10, -1, np.int16)
        if goal is not None:
            first[goal] = T
        act = g.normal(size=(T, 7)) * 0.3
        ent = g.uniform(3, 4, size=(T, 7)) + out_signal * bad
        recs.append(dict(kind=kind, ntype=ntype, task=task, init=init, T=T, success=goal == task and kind == "clean",
                         first_goal=first, a0e=act[0] + g.normal(size=7), act=act, ent=ent, var=g.uniform(size=(T, 7)) * 0.01,
                         hid=hid.astype(np.float16)))

    for t in range(10):
        for i in range(20):
            f = g.random() < 0.25 + 0.02 * t
            mk("clean", t, f, i, goal=None if f else t)
    for nt in vk.NONTASK:
        for t in range(10):
            for i in range(2):
                u = g.random() < 0.5
                mk("nontask", t, False, 20 + i, nt, shift=2.5 * (hid_signal > 0), goal=(t + 1) % 10 if u else None)
    for t in range(9):
        for i in range(6):
            red = g.random() < 0.3
            mk("attack", t, False, 23 + i, shift=2.5 * red * (hid_signal > 0), goal=vk.TARGET if red else None)
    return recs


def test_analysis_planted_pass():
    probe.N_BOOT = 200
    m = probe.analyze(synth(), log=lambda *a: None)
    assert m["valid"], m["validity"]
    assert m["probe_auc"][0] > 0.9 and m["best_base_auc"][0] < 0.7 and abs(np.mean(m["shuffled"]) - 0.5) < 0.07
    assert m["transfer_auc"][0] > 0.8 and m["gate_probe"]["reduction"] > 0.8 and m["gate_probe"]["success_cost"] <= 0.05
    assert m["overall"] == "PASS" and m["attack"]["label"] == "YES"


def test_analysis_output_signal_fail():
    probe.N_BOOT = 200
    m = probe.analyze(synth(seed=1, hid_signal=0.0, out_signal=1.0), log=lambda *a: None)
    assert m["best_base_auc"][0] > 0.9 and m["kill"] and m["overall"] == "FAIL"


def test_verdict_rules():
    base = dict(valid=True, probe_auc=[0.85, 0.80, 0.90], best_base_auc=[0.70, 0.64, 0.76], transfer_auc=[0.85, 0.8, 0.9],
                gate_probe=dict(reduction=0.6, success_cost=0.04), gate_base_best_reduction=0.3)
    assert probe.verdict(base)[0] == "PASS"
    assert probe.verdict(dict(base, probe_auc=[0.85, 0.75, 0.90]))[0] == "MIXED"          # CIs overlap
    assert probe.verdict(dict(base, probe_auc=[0.72, 0.66, 0.78]))[0] == "FAIL"           # <= best + 0.03
    assert probe.verdict(dict(base, transfer_auc=[0.79, 0.7, 0.85]))[0] == "MIXED"
    assert probe.verdict(dict(base, gate_probe=dict(reduction=0.6, success_cost=0.06)))[0] == "MIXED"
    assert probe.verdict(dict(base, gate_base_best_reduction=0.7))[0] == "MIXED"
    assert probe.verdict(dict(base, valid=False))[0] == "INVALID"


# ------------------------------------------------------------------------------------------------ GPU pre-flight
@pytest.fixture(scope="module")
def pol():
    return vk.Policy()


@pytest.fixture(scope="module")
def env():
    return vk.LiberoEnv()


@gpu
def test_gpu_env_goals(env):
    imgs = []
    for t in range(10):
        img, g = env.start(t, 0)
        assert img.shape == (224, 224, 3) and g.shape == (10,) and not g[t]
        imgs.append(img)
    img2, _ = env.step([0.5, 0, 0, 0, 0, 0, -1])
    for _ in range(5):
        img2, _ = env.step([0.5, 0, 0, 0, 0, 0, -1])
    assert np.abs(img2.astype(int) - imgs[-1]).mean() > 0.5        # the arm moved in the rendered image


@gpu
def test_gpu_decode_matches_predict_action(pol, env):
    from PIL import Image
    imgs = np.stack([env.start(t, i)[0] for t, i in [(0, 0), (1, 0), (2, 1), (3, 2)]])
    inp = pol.proc(vk.prompt(vk.LANG[0]), Image.fromarray(imgs[0])).to("cuda", dtype=torch.bfloat16)
    assert torch.allclose(inp["pixel_values"].float(), pol.pixels(pol.to01(imgs[:1])).float(), atol=2e-2)
    ref = pol.vla.predict_action(**inp, unnorm_key=pol.unnorm_key, do_sample=False)
    one = pol.act(imgs[:1], vk.LANG[0])
    assert np.allclose(vk.unnorm(one["norm"][0], pol.stats), ref, atol=1e-5), (vk.unnorm(one["norm"][0], pol.stats), ref)
    assert one["hid"].shape == (1, len(vk.LAYERS), 4096) and np.isfinite(one["hid"]).all()
    single = np.concatenate([pol.act(imgs[i:i + 1], vk.LANG[0], hidden=False)["toks"] for i in range(4)])
    batch = pol.act(imgs, vk.LANG[0])
    assert (batch["toks"] == single).mean() >= 0.9, (batch["toks"], single)
    print(f"[pre-flight] batched/single token agreement {(batch['toks'] == single).mean():.3f}")


@gpu
def test_gpu_patch_grad(pol, env):
    fr = np.stack([env.start(t, 5)[0] for t in (0, 1, 2, 3)])
    patch, hist = pol.train_patch(fr, steps=2, log=lambda *a: None)
    torch.manual_seed(0)
    init = (torch.rand(3, vk.PATCH, vk.PATCH, device="cuda").permute(1, 2, 0) * 255).round().byte().cpu().numpy()
    assert patch.shape == (vk.PATCH, vk.PATCH, 3) and np.isfinite(hist).all() and (patch != init).mean() > 0.5


@gpu
def test_gpu_pool_and_group(pol):
    pool = vk.EnvPool(2)
    try:
        g = dict(key="pf", instr=vk.LANG[0], episodes=[dict(kind="clean", task=0, init=0), dict(kind="attack", task=0, init=23),
                                                     dict(kind="nontask", ntype="empty", task=1, init=20)])
        recs, timing = vk.run_group(pol, pool, g, np.zeros((vk.PATCH, vk.PATCH, 3), np.uint8), max_steps=9)
    finally:
        pool.close()
    for r in recs:
        assert r["T"] == 9 and r["act"].shape == (9, 7) and r["hid"].shape == (2, len(vk.LAYERS), 4096) and r["a0e"].shape == (7,)
    print(f"[pre-flight] sec/step (batch, s): {timing[:3]}; GPU mem peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
