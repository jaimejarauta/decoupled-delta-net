"""correction-memory: a gradient-free activation-space memory of ONE corrective demo for SmolVLA on LIBERO.

Memory write (no weight update): run the frozen action expert along the straight flow path noise -> actions twice on the
demo's frames, once with the policy's OWN predicted chunk and once with the DEMO chunk (teacher-forced), and store
d = mean(h_demo - h_ref) at one expert site under a whitened key of the episode's first-frame prefix embedding ("gist").
Memory read: at the start of a new episode, if the gist matches a stored key (cosine > tau), alpha * d is added at the
site for every denoising step of the episode.
"""
import contextlib, json, math, os, shutil, tempfile

import numpy as np
import torch
from torch import nn

REPO, REV = "HuggingFaceVLA/smolvla_libero", "6721902bc4d61e50a3bfdb11dfb4cb626f05d102"
VLM_ID, VLM_REV = "HuggingFaceTB/SmolVLM2-500M-Instruct", "7b375e1b73b11138ff12fe22c8f2822d8fe03467"
DATA = "lerobot/libero"
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
N_ACTION_STEPS = 10          # replan every 10 steps (checkpoint default 1 is 10x slower; pi05 LIBERO uses 10)
REPHRASE = "carefully {}"    # the one instruction-rephrasing baseline, fixed in advance (not tuned)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
A_DIM = 7


# ----------------------------------------------------------------------------------------------- loading
def load_policy():
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    p = SmolVLAPolicy.from_pretrained(REPO, revision=REV)
    p.config.n_action_steps = N_ACTION_STEPS
    p.to(DEV).eval()
    pre, post = make_pre_post_processors(p.config, pretrained_path=REPO, pretrained_revision=REV,
                                         preprocessor_overrides={"device_processor": {"device": DEV}})
    return p, pre, post


def tiny_policy(seed=0):
    """Random tiny SmolVLA (same code path, ~1M params) for CPU tests. Only small json/txt files are downloaded."""
    from huggingface_hub import snapshot_download
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    src = snapshot_download(VLM_ID, revision=VLM_REV, allow_patterns=["*.json", "*.txt"])
    d = tempfile.mkdtemp(prefix="tiny_smolvlm_")
    for f in os.listdir(src):
        shutil.copy(os.path.join(src, f), d)
    c = json.load(open(os.path.join(d, "config.json")))
    c["vision_config"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, patch_size=32)
    c["text_config"].update(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16)
    json.dump(c, open(os.path.join(d, "config.json"), "w"))
    torch.manual_seed(seed)
    vis = lambda: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64))
    cfg = SmolVLAConfig(vlm_model_name=d, load_vlm_weights=False, device="cpu", n_action_steps=N_ACTION_STEPS,
                        num_vlm_layers=0, expert_width_multiplier=0.5,
                        input_features={"observation.images.image": vis(), "observation.images.image2": vis(),
                                        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,))},
                        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(A_DIM,))})
    p = SmolVLAPolicy(cfg).eval()
    stats = {"observation.state": {"mean": torch.zeros(8), "std": torch.ones(8)},
             "action": {"mean": torch.zeros(A_DIM), "std": torch.ones(A_DIM)}}
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    return p, pre, post


def sites(policy):
    """'out' = input of action_out_proj (final normed expert state, linear map to velocity);
    'resJ' = expert residual stream after layer J = n_layers - 4 (late but not last)."""
    return ("out", f"res{len(policy.model.vlm_with_expert.lm_expert.layers) - 4}")


# ----------------------------------------------------------------------------------------------- key whitening
class Whitener:
    """Ridge PCA-whitening of gist keys over a reference pool: each pool principal direction is scaled by
    1/sqrt(var + mean(var)) (shrinkage 1; plain whitening of N pool keys makes them exactly equidistant), then unit
    norm -> cosine similarity."""

    def __init__(self, keys):
        x = keys.double().cpu()   # keys come from the GPU; __call__ works on CPU
        self.mu = x.mean(0)
        _, s, vt = torch.linalg.svd(x - self.mu, full_matrices=False)
        k = int((s > 1e-6 * s[0]).sum())
        var = s[:k] ** 2 / max(len(x) - 1, 1)
        self.U, self.s = vt[:k].T, (var + var.mean()).sqrt()

    def __call__(self, key):
        z = ((key.double().cpu() - self.mu) @ self.U) / self.s
        return z / z.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def pick_tau(z, labels):
    """Threshold = midpoint between the 95th pct of different-label cosines and the 5th pct of same-label cosines."""
    sim = z @ z.T
    lab = torch.as_tensor(labels)
    same = (lab[:, None] == lab[None]) & ~torch.eye(len(lab), dtype=torch.bool)
    diff = lab[:, None] != lab[None]
    return float(0.5 * (torch.quantile(sim[diff].float(), 0.95) + torch.quantile(sim[same].float(), 0.05)))


# ----------------------------------------------------------------------------------------------- hooks
class Memory:
    """All hooks on one policy. mode: 'off' | 'gated' (bank + whitener + tau, decided at the episode's first chunk)
    | 'always' (self.vec for every element) | 'raw' (self.add given directly, used for Jacobians)."""

    def __init__(self, policy):
        self.p, self.sites = policy, sites(policy)
        self.mode, self.site, self.alpha = "off", self.sites[0], 0.0
        self.bank, self.whiten, self.tau, self.vec = [], None, None, None   # bank: list of (z_key (k,), vec (H,))
        self.add = None             # (B, H) currently injected at self.site
        self.cap = None             # {site: [ (B,H) ... ]} while capturing
        self.fired = None           # (B,) bank index fired this episode, -1 = none
        self.last_key = None
        m = policy.model
        m.action_out_proj.register_forward_pre_hook(self._out)
        j = int(self.sites[1][3:])
        layer = m.vlm_with_expert.lm_expert.layers[j]
        layer.post_attention_layernorm.register_forward_pre_hook(self._res_in)
        layer.mlp.register_forward_hook(self._res_out)
        embed = m.embed_prefix

        def embed_prefix(*a, **k):
            out = embed(*a, **k)
            self._on_prefix(out, a[0], a[3] if len(a) > 3 else k["lang_masks"])
            return out
        m.embed_prefix = embed_prefix
        reset = policy.reset

        def new_reset():
            reset()
            self.fired = None
            if self.mode != "raw":
                self.add = None
        policy.reset = new_reset

    # --- injection + capture at the two sites (capture records the post-injection state)
    def _inject(self, site, h):
        if self.add is not None and self.mode != "off" and site == self.site:
            h = h + self.add[:, None, :].to(h.dtype)
        return h

    def _out(self, mod, args):
        h = self._inject("out", args[0])
        if self.cap is not None:
            self.cap["out"].append(h.detach().float().mean(1))
        return (h,) + tuple(args[1:])

    def _res_in(self, mod, args):
        self._r = args[0]

    def _res_out(self, mod, args, out):
        out = self._inject(self.sites[1], out)      # adding to the MLP output = adding to the residual after the layer
        if self.cap is not None:
            self.cap[self.sites[1]].append((self._r + out).detach().float().mean(1))
        return out

    # --- gist key + gating
    def _on_prefix(self, out, images, lang_masks):
        embs = out[0]
        k, L = len(images), lang_masks.shape[1]
        n = (embs.shape[1] - L - 1) // k            # prefix = k images x n tokens, L language tokens, 1 state token
        img = embs[:, :k * n].float().mean(1)
        lm = lang_masks[:, :, None].float()
        lang = (embs[:, k * n:k * n + L].float() * lm).sum(1) / lm.sum(1).clamp_min(1)
        self.last_key = torch.cat([img, lang], -1)
        if self.mode == "gated" and self.fired is None:
            z = self.whiten(self.last_key)
            Z = torch.stack([b[0] for b in self.bank])
            sim = z @ Z.T
            best, ok = sim.argmax(1), sim.max(1).values > self.tau
            self.fired = torch.where(ok, best, torch.full_like(best, -1))
            V = torch.stack([b[1] for b in self.bank]).to(embs.device)
            self.add = self.alpha * V[best] * ok.to(embs.device)[:, None]
        elif self.mode == "always":
            self.add = self.alpha * self.vec.to(embs.device)[None].expand(embs.shape[0], -1)

    @contextlib.contextmanager
    def using(self, mode="off", bank=None, alpha=0.0, site=None, vec=None):
        old = (self.mode, self.bank, self.alpha, self.site, self.vec, self.add, self.fired)
        self.mode, self.bank, self.alpha, self.vec = mode, bank or [], alpha, vec
        self.site, self.add, self.fired = site or self.site, None, None
        try:
            yield self
        finally:
            self.mode, self.bank, self.alpha, self.site, self.vec, self.add, self.fired = old


# ----------------------------------------------------------------------------------------------- teacher forcing
def _model_inputs(policy, batch):
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    images, img_masks = policy.prepare_images(batch)
    return images, img_masks, batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK], policy.prepare_state(batch)


def own_chunk(policy, batch, noise):
    """Policy's own normalized, padded (B, chunk, 32) action chunk for this noise."""
    return policy.model.sample_actions(*_model_inputs(policy, batch), noise=noise)


@torch.no_grad()
def site_states(policy, mem, batch, actions, noise):
    """{site: (B, H)}: site state averaged over the num_steps Euler times and the chunk positions, with the expert
    driven along the straight path x_t = t*noise + (1-t)*actions (NOISE_AT_ONE convention, t = 1 - k/N)."""
    from lerobot.policies.smolvla import modeling_smolvla as ms
    m = policy.model
    assert m.config.use_cache
    images, img_masks, lt, lm, state = _model_inputs(policy, batch)
    pe, ppm, pam = m.embed_prefix(images, img_masks, lt, lm, state=state)
    _, kv = m.vlm_with_expert.forward(attention_mask=ms.make_att_2d_masks(ppm, pam), position_ids=torch.cumsum(ppm, 1) - 1,
                                      past_key_values=None, inputs_embeds=[pe, None], use_cache=True)
    mem.cap = {s: [] for s in mem.sites}
    N, B = m.config.num_steps, actions.shape[0]
    for k in range(N):
        t = 1.0 - k / N
        m.denoise_step(ppm, kv, t * noise + (1 - t) * actions, torch.full((B,), t, device=actions.device))
    out = {s: torch.stack(v).mean(0) for s, v in mem.cap.items()}
    mem.cap = None
    return out


def demo_shift(policy, mem, batch, seed=0):
    """Per-frame site states for the demo chunk and for the policy's own chunk (same noise).
    Returns (S_demo {site:(B,H)}, S_own {site:(B,H)}, a_own (B,chunk,32))."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a_dem = policy.prepare_action(batch)
    noise = torch.randn(a_dem.shape, generator=g).to(a_dem.device, a_dem.dtype)
    with mem.using("off"), torch.no_grad():
        a_own = own_chunk(policy, batch, noise)
        s_dem = site_states(policy, mem, batch, a_dem, noise)
        s_own = site_states(policy, mem, batch, a_own, noise)
    return s_dem, s_own, a_own


def gn_sharp_dirs(policy, mem, batch, site, r=8, probes=32, seed=0):
    """Top-r eigenvectors of the Gauss-Newton matrix G = sum_frames J^T J, J = d(action chunk)/d(shift at site) at 0,
    estimated with `probes` random vector-Jacobian products per frame (G_hat = E[J^T u u^T J], u ~ N(0, I)).
    ponytail: randomized GN estimate of the steering Hessian (MSE behavioural loss); exact Jacobian if r ~ probes."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = policy.prepare_action(batch)
    B, H = a.shape[0], policy.model.vlm_with_expert.expert_hidden_size
    noise = torch.randn(a.shape, generator=g).to(a.device, a.dtype)
    delta = torch.zeros(B, H, device=a.device, requires_grad=True)
    G = torch.zeros(H, H, dtype=torch.float64)
    with mem.using("raw", site=site):
        mem.add = delta
        with torch.enable_grad():
            out = own_chunk(policy, batch, noise)[:, :, :A_DIM]
            for _ in range(probes):
                u = torch.randn(out.shape, generator=g).to(out.device)
                (gr,) = torch.autograd.grad((out * u).sum(), delta, retain_graph=True)
                gr = gr.double().cpu()
                G += gr.T @ gr / probes
    evals, evecs = torch.linalg.eigh(G)
    return evecs[:, -r:].flip(-1).float(), evals.flip(-1).float()


def remove_dirs(d, V):
    return d - V @ (V.T @ d)


# ----------------------------------------------------------------------------------------------- LoRA baseline
class LoRALinear(nn.Module):
    def __init__(self, base, r=16, alpha=16):
        super().__init__()
        self.base, self.scale = base, alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))

    @property
    def weight(self):                      # lerobot reads q_proj.weight.dtype
        return self.base.weight

    def forward(self, x):
        return self.base(x) + (x.to(self.A.dtype) @ self.A.T @ self.B.T * self.scale).to(x.dtype)


def _expert_linears(policy):
    ex = policy.model.vlm_with_expert.lm_expert
    return [(n, m) for n, m in ex.named_modules() if isinstance(m, nn.Linear)], ex


def lora_finetune(policy, batch, lr, steps=200, bs=16, r=16, seed=0):
    """LoRA (rank r) on every linear layer of the action expert, trained with the policy's own flow-matching loss on
    the frames of ONE demo. Returns {param_name: delta_W} (merged LoRA update); the policy is left unchanged."""
    torch.manual_seed(seed)
    lins, ex = _expert_linears(policy)
    wraps = {}
    for name, lin in lins:
        parent = ex.get_submodule(name.rsplit(".", 1)[0]) if "." in name else ex
        w = LoRALinear(lin, r)
        setattr(parent, name.rsplit(".", 1)[-1], w)
        wraps[name] = (parent, w)
    params = [q for _, w in wraps.values() for q in (w.A, w.B)]
    frozen = [q for q in policy.parameters() if q.requires_grad and not any(q is z for z in params)]
    for q in frozen:
        q.requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n = batch["action"].shape[0]
    pick = lambda v, i: v[i.to(v.device)] if torch.is_tensor(v) and v.shape[:1] == (n,) else (
        [v[j] for j in i.tolist()] if isinstance(v, list) and len(v) == n else v)
    losses = []
    try:
        with torch.enable_grad():
            for _ in range(steps):
                idx = torch.randint(0, n, (min(bs, n),))
                loss, _ = policy.forward({k: pick(v, idx) for k, v in batch.items()})
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        deltas = {f"{nm}.weight": (w.B @ w.A * w.scale).detach().clone() for nm, (_, w) in wraps.items()}
    finally:
        for name, (parent, w) in wraps.items():
            setattr(parent, name.rsplit(".", 1)[-1], w.base)
        for q in frozen:
            q.requires_grad_(True)
    return deltas, losses


@contextlib.contextmanager
def lora_applied(policy, deltas):
    ex = policy.model.vlm_with_expert.lm_expert
    orig = {k: ex.get_parameter(k).detach().clone() for k in deltas}
    with torch.no_grad():
        for k, dW in deltas.items():
            ex.get_parameter(k).add_(dW.to(ex.get_parameter(k).dtype))
    try:
        yield
    finally:
        with torch.no_grad():
            for k, w in orig.items():
                ex.get_parameter(k).copy_(w)


# ----------------------------------------------------------------------------------------------- rollouts
class Rephrased:
    """Wraps the policy preprocessor so the task instruction is rephrased (instruction-rephrasing baseline)."""

    def __init__(self, pre, template=REPHRASE):
        self.pre, self.t = pre, template

    def __call__(self, obs):
        return self.pre({**obs, "task": [self.t.format(s) for s in obs["task"]]})

    def reset(self):
        self.pre.reset()


def _seed_selects_init(make):
    """Env factory wrapper (runs inside the env worker): reset(seed) starts from LIBERO init state seed % 1000.
    lerobot 0.7 LocalVectorEnv has no set_attr, and LiberoEnv otherwise walks init states with a fixed stride."""
    env = make()
    reset = env.reset

    def seeded_reset(seed=None, **kw):
        if seed is not None:
            env.init_state_id = int(seed) % 1000
        return reset(seed=seed, **kw)
    env.reset = seeded_reset
    return env


def make_task_env(suite, tid, n):
    from functools import partial
    from lerobot.envs.configs import LiberoEnvConfig
    cfg = LiberoEnvConfig(task=suite, task_ids=[tid])
    vec = cfg.create_envs(n_envs=n, use_async_envs=n > 1)[suite][tid]
    # ponytail: patches private LocalVectorEnv fields (lerobot pinned ==0.7.0); rebuilt lazily on the next reset
    vec._close_envs()
    vec._env_fns = [partial(_seed_selects_init, fn) for fn in vec._env_fns]
    return vec, cfg


def run_episodes(vec, policy, procs, init_ids, seed):
    """One batched rollout: sub-env i starts from LIBERO init state init_ids[i]. Returns per-episode success list."""
    from lerobot.scripts.lerobot_eval import rollout
    envpre, envpost, pre, post = procs
    assert seed % 1000 == 0 and all(0 <= int(i) < 1000 for i in init_ids), (seed, init_ids)  # seed encodes init id
    torch.manual_seed(seed)
    np.random.seed(seed)
    r = rollout(vec, policy, envpre, envpost, pre, post, seeds=[seed + int(i) for i in init_ids])
    done, succ = r["done"], r["success"]
    last = done.int().argmax(1)
    return [bool(succ[b, : last[b] + 1].any()) for b in range(len(init_ids))]


# ----------------------------------------------------------------------------------------------- stats
def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan")] * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [p, max(0.0, c - h), min(1.0, c + h)]


def paired_boot(a, b, n_boot=10000, seed=0):
    """Mean of a - b over paired episodes with a 95% percentile bootstrap CI. a, b: equal-length 0/1 lists."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if len(d) == 0:
        return [float("nan")] * 3 + [0]
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return [float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), int(len(d))]


# ----------------------------------------------------------------------------------------------- demos (LIBERO human demos)
def load_dataset():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(DATA)
    ds = LeRobotDataset(DATA, delta_timestamps={"action": [i / meta.fps for i in range(50)]})
    return ds


def episodes_by_task(ds):
    """{task string: [(episode_index, from_index, to_index), ...]} from the dataset metadata."""
    e = ds.meta.episodes
    if "tasks" not in e.column_names:   # lerobot/libero v3.0 episodes parquet has no "tasks": use frame task_index
        t = ds.hf_dataset.data   # raw arrow table (no torch transform)
        frame_task = dict(zip(t.column("index").to_pylist(), t.column("task_index").to_pylist()))
        name = {int(i): s for s, i in zip(ds.meta.tasks.index, ds.meta.tasks["task_index"])}
    out = {}
    for i in range(len(e)):
        r = e[i]
        task = r["tasks"][0] if "tasks" in r else name[frame_task[int(r["dataset_from_index"])]]
        out.setdefault(task, []).append((int(r["episode_index"]), int(r["dataset_from_index"]), int(r["dataset_to_index"])))
    return out


KEEP = ("observation.images.image", "observation.images.image2", "observation.state", "action", "action_is_pad", "task")


def frames_batch(ds, idxs, pre):
    """Preprocessed policy batch for dataset frames `idxs` (same pipeline as at eval time, plus the action chunk)."""
    from torch.utils.data import default_collate
    raw = default_collate([{k: it[k] for k in KEEP} for it in (ds[int(i)] for i in idxs)])
    pad = raw.pop("action_is_pad")
    b = pre(raw)
    b["action_is_pad"] = pad.to(b["action"].device)
    return b
