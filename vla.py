"""SmolVLA (lerobot/smolvla_libero) + LIBERO-Goal glue: closed-loop rollouts with a steering hook, state collection,
CAA vectors, behavioural action losses and their steering Hessian. GPU path; exercised by the Colab pre-flight."""
import os
from contextlib import nullcontext
import numpy as np
import torch
from steer import Steer, fd_hvp, lanczos

REPO = os.environ.get("SSV_POLICY", "lerobot/smolvla_libero")
SUITE = "libero_goal"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H_EXEC = 10                      # n_action_steps (executed part of each 50-step chunk), frozen
torch.backends.cuda.matmul.allow_tf32 = False   # fp32 everywhere (finite-difference HVPs need it)
torch.backends.cudnn.allow_tf32 = False

# Units (frozen): task switch (TS) = instruction A given, success of task B scored in B's env;
# start suppression (SS) = task instruction given, target = robot stays still.
UNITS = {
    "TS1": dict(kind="TS", A="put the bowl on the stove", B="put the bowl on the plate"),
    "TS2": dict(kind="TS", A="open the middle drawer of the cabinet", B="turn on the stove"),
    "SS1": dict(kind="SS", A="put the bowl on the stove"),
    "SS2": dict(kind="SS", A="open the middle drawer of the cabinet"),
}
SS_POS = ["", "do nothing", "stay still", "do not move"]   # CAA positive set for start suppression
SS_STEPS, TS_STEPS = 100, 300
SS_THRESH = 0.02   # metres: max EEF displacement over the first 100 steps below this = start suppressed


def task_id(language):
    from libero.libero import benchmark
    s = benchmark.get_benchmark_dict()[SUITE]()
    ids = [i for i in range(s.n_tasks) if s.get_task(i).language.strip().lower() == language]
    if len(ids) != 1:
        raise ValueError(f"task {language!r} not found once in {SUITE}: {[s.get_task(i).language for i in range(s.n_tasks)]}")
    return ids[0]


def env_task(unit):
    u = UNITS[unit]
    return task_id(u["B"] if u["kind"] == "TS" else u["A"])


def load_policy():
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies import make_pre_post_processors
    pol = SmolVLAPolicy.from_pretrained(REPO).to(DEV).float().eval()
    pol.requires_grad_(False)
    pol.config.n_action_steps = H_EXEC
    pol.reset()
    pre, post = make_pre_post_processors(pol.config, pretrained_path=REPO,
                                         preprocessor_overrides={"device_processor": {"device": DEV}})
    return pol, pre, post


def text_layers(pol):
    return pol.model.vlm_with_expert.get_vlm_model().text_model.layers


def _batch(pre, states, instr):
    from lerobot.envs.utils import prepare_canonical_observation
    b = prepare_canonical_observation({k: np.stack([s[k] for s in states]) for k in states[0]})
    b["task"] = [instr] * len(states)
    return pre(b)


def make_steer(pol, L, vec, alpha):
    if vec is None or alpha == 0:
        return nullcontext()
    return Steer(text_layers(pol)[L], torch.as_tensor(vec, dtype=torch.float32, device=DEV), alpha)


def rollout(pol, pre, post, pool, init_ids, instr, steer_ctx, max_steps, seed, record_t=()):
    """One batch of closed-loop episodes (one per init id) with a fixed instruction and steering context.
    Same seed + same init ids across arms -> paired comparison (common random numbers for the flow noise)."""
    from lerobot.processor import policy_output_to_transition, transition_to_policy_action
    pol.reset(); pre.reset(); post.reset()
    torch.manual_seed(seed)
    obs = pool.reset(init_ids)
    B = len(init_ids)
    done, succ = np.zeros(B, bool), np.zeros(B, bool)
    p0 = np.stack([o["observation.state"][:3] for o in obs])
    maxd, maxd50, steps = np.zeros(B), np.zeros(B), np.zeros(B, int)
    rec = []
    for t in range(max_steps):
        if t in record_t:
            rec += [obs[i] for i in range(B) if not done[i]]
        b = _batch(pre, obs, instr)
        with torch.inference_mode(), steer_ctx:
            a = transition_to_policy_action(policy_output_to_transition(pol.select_action(b)))
        a = post(a).to("cpu").numpy()
        run = np.where(~done)[0]
        res = pool.step({int(i): a[i] for i in run})
        for i in run:
            obs[i], s, term = res[int(i)]
            succ[i] |= s
            done[i] |= term or s
            steps[i] = t + 1
            maxd[i] = max(maxd[i], float(np.linalg.norm(obs[i]["observation.state"][:3] - p0[i])))
        if t == 49:
            maxd50 = maxd.copy()
        if done.all():
            break
    return dict(success=succ.tolist(), maxdisp=maxd.tolist(), maxdisp50=maxd50.tolist(), steps=steps.tolist()), rec


def chunk(pol, pre, states, instr, noise, steer_ctx=None):
    """Normalized action chunk [B, 50, 7] for fixed flow noise (deterministic, differentiable wrt the steering vec)."""
    b = _batch(pre, states, instr)
    with steer_ctx or nullcontext():
        return pol._get_action_chunk(pol._prepare_batch(b), noise=noise)


def fixed_noise(pol, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, pol.config.chunk_size, pol.config.max_action_dim, generator=g).to(DEV)


def residual_means(pol, pre, states, instr, L, mb=8):
    """Mean (over valid prefix positions) of the residual stream after text layer L, per state -> [n, d]."""
    out, orig = [], pol.model.embed_prefix
    store = {}

    def ep(*a, **k):
        r = orig(*a, **k)
        store["mask"] = r[1]
        return r
    pol.model.embed_prefix = ep
    try:
        for i in range(0, len(states), mb):
            s = states[i:i + mb]
            with torch.no_grad(), Steer(text_layers(pol)[L], record=True) as st:
                chunk(pol, pre, s, instr, fixed_noise(pol, len(s)))
            m = store["mask"].float()[..., None]
            out.append(((st.h[0].float() * m).sum(1) / m.sum(1)).cpu())
    finally:
        del pol.model.embed_prefix   # back to the class method
    return torch.cat(out)


def caa_vector(pol, pre, states, unit, L):
    u = UNITS[unit]
    neg = residual_means(pol, pre, states, u["A"], L).mean(0)
    pos = [u["B"]] if u["kind"] == "TS" else SS_POS
    return torch.stack([residual_means(pol, pre, states, p, L).mean(0) for p in pos]).mean(0) - neg


def post_affine(post):
    """raw = s * normalized + b, measured from the checkpoint's postprocessor (MEAN_STD unnormalizer)."""
    z = post(torch.zeros(1, 7, device=DEV)).cpu().double()[0]
    o = post(torch.ones(1, 7, device=DEV)).cpu().double()[0]
    return (o - z), z


class Loss:
    """Behavioural loss L(w) over a fixed set of states, w = steering vector added at layer L.
    TS: MSE between the executed steps of the steered chunk (instruction A) and the unsteered chunk for instruction B.
    SS: MSE between the executed motion dims (0-5) and the normalized no-op (zero delta) action."""

    def __init__(self, pol, pre, post, states, unit, L, mb=8):
        self.pol, self.pre, self.states, self.L, self.mb = pol, pre, states, L, mb
        self.u = UNITS[unit]
        self.noise = fixed_noise(pol, len(states), seed=123)
        if self.u["kind"] == "TS":
            with torch.no_grad():
                self.target = torch.cat([chunk(pol, pre, states[i:i + mb], self.u["B"], self.noise[i:i + mb])
                                         for i in range(0, len(states), mb)])[:, :H_EXEC, :7]
        else:
            s, b = post_affine(post)
            self.target = ((0 - b[:6]) / s[:6]).float().to(DEV)

    def per_state(self, w, alpha=1.0, grad=False):
        w = torch.as_tensor(w, dtype=torch.float32, device=DEV)
        if grad:
            w = w.detach().requires_grad_(True)
        losses = []
        for i in range(0, len(self.states), self.mb):
            ctx = Steer(text_layers(self.pol)[self.L], w, alpha) if alpha else nullcontext()
            with torch.set_grad_enabled(grad):
                a = chunk(self.pol, self.pre, self.states[i:i + self.mb], self.u["A"], self.noise[i:i + self.mb], ctx)
            a = a[:, :H_EXEC]
            if self.u["kind"] == "TS":
                l = ((a[..., :7] - self.target[i:i + self.mb]) ** 2).mean((1, 2))
            else:
                l = ((a[..., :6] - self.target) ** 2).mean((1, 2))
            if grad:
                (l.sum() / len(self.states)).backward()
            losses.append(l.detach())
        return torch.cat(losses), (w.grad.detach().double().cpu() if grad else None)

    def grad(self, w):
        return self.per_state(w, grad=True)[1]


def steering_hessian(loss, w, m=40, k=15, rel_delta=0.01):
    """Top-k sharp directions of H = d2L/dw2 at the operating point w (Lanczos on finite-difference HVPs)."""
    w = torch.as_tensor(w, dtype=torch.float64)
    delta = rel_delta * max(float(w.norm()), 1.0)
    return lanczos(lambda u: fd_hvp(loss.grad, w, u, delta), w.numel(), m=m, k=k)
