"""vla-knew rollouts: OpenVLA-7B (LIBERO-Goal finetune) in LIBERO-Goal. Logs, per policy step, the greedy action and the
output-space features (token entropy, token-distribution variance); at every chunk boundary (every CHUNK steps) the
last-prompt-token residual stream at LAYERS. Eval protocol mirrors openvla/experiments/robot/libero/run_libero_eval.py.
GPU + LIBERO paths are exercised by the Colab pre-flight (test_vk.py with VK_PREFLIGHT=1)."""
import os, io, math, time
import numpy as np
from PIL import Image

MODEL = "openvla/openvla-7b-finetuned-libero-goal"
SUITE = "libero_goal"
TASK_NAMES = ["open_the_middle_drawer_of_the_cabinet", "put_the_bowl_on_the_stove", "put_the_wine_bottle_on_top_of_the_cabinet",
              "open_the_top_drawer_and_put_the_bowl_inside", "put_the_bowl_on_top_of_the_cabinet",
              "push_the_plate_to_the_front_of_the_stove", "put_the_cream_cheese_in_the_bowl", "turn_on_the_stove",
              "put_the_bowl_on_the_plate", "put_the_wine_bottle_on_the_rack"]  # LIBERO task-map order
LANG = [n.replace("_", " ") for n in TASK_NAMES]
NONTASK = {"empty": "", "nonsense": "glorp the vintage seven quietly under",
           "infeasible": "put the stove inside the bowl", "wrongtask": "pick up the alphabet soup and place it in the basket"}
TARGET = 8                      # attacker's task: "put the bowl on the plate"
MAX_STEPS, WAIT, CHUNK = 300, 10, 8
LAYERS = (4, 8, 12, 16, 20, 24, 28, 32)   # indices into hidden_states (0 = embeddings, 32 = last layer)
V_ACT0, V_END = 31744, 32000    # the 256 action tokens (OpenVLA: discretized = 32000 - id)
_b = np.linspace(-1, 1, 256)
BIN_CENTERS = (_b[:-1] + _b[1:]) / 2
TOKEN_VALUE = BIN_CENTERS[np.clip(V_END - np.arange(V_ACT0, V_END) - 1, 0, 254)]
DUMMY = [0, 0, 0, 0, 0, 0, -1]
PATCH, PATCH_YX = 48, (4, 4)    # digital patch on the 224x224 model input (top-left = back wall / table edge)


def prompt(instr):
    return f"In: What action should the robot take to {instr.lower()}?\nOut:"


def prep(agentview):
    """256x256 agentview -> 224x224 model image: rotate 180, JPEG round trip, lanczos resize, center crop (area 0.9) and
    bilinear resize back (openvla LIBERO eval, re-implemented with PIL instead of TF)."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(agentview[::-1, ::-1])).save(buf, "JPEG", quality=95)
    im = Image.open(buf).convert("RGB").resize((224, 224), Image.LANCZOS)
    s = math.sqrt(0.9) * 224
    o = (224 - s) / 2
    return np.asarray(im.resize((224, 224), Image.BILINEAR, box=(o, o, o + s, o + s)))


def unnorm(norm, stats):
    """= the unnormalization in OpenVLA predict_action (q01/q99, masked dims)."""
    q01, q99 = np.asarray(stats["q01"]), np.asarray(stats["q99"])
    mask = np.asarray(stats.get("mask", np.ones_like(q01, dtype=bool)))
    return np.where(mask, 0.5 * (norm + 1) * (q99 - q01) + q01, norm)


def env_action(norm, stats):
    """Normalized [-1,1] action -> env action (unnormalize, binarize gripper to [-1,1], invert gripper sign)."""
    a = unnorm(norm, stats)
    a[..., -1] = -np.sign(2 * a[..., -1] - 1)
    return a


def out_feats(L):
    """L: [B, 7, V] logits of the 7 greedy action tokens -> greedy tokens, normalized action, per-dim entropy and variance
    of the action-token distribution (analytic stand-in for sampling variance, same forward pass)."""
    import torch
    L = L.float()
    toks = L.argmax(-1)
    norm = BIN_CENTERS[np.clip(V_END - toks.cpu().numpy() - 1, 0, 254)].astype(np.float32)
    lp = L[..., V_ACT0:V_END].log_softmax(-1)
    p = lp.exp()
    val = torch.as_tensor(TOKEN_VALUE, dtype=torch.float32, device=L.device)
    m = (p * val).sum(-1)
    return dict(toks=toks.cpu().numpy(), norm=norm, ent=(-(p * lp).sum(-1)).cpu().numpy(),
                var=((p * val ** 2).sum(-1) - m ** 2).clamp_min(0).cpu().numpy())


def paste(imgs, patch, mask):
    """imgs [B,224,224,3] uint8; patch [P,P,3] uint8; mask [B] bool -> copy with the patch on masked images."""
    imgs = imgs.copy()
    y, x = PATCH_YX
    imgs[np.asarray(mask, bool), y:y + PATCH, x:x + PATCH] = patch
    return imgs


class Policy:
    def __init__(self, name=MODEL):
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                                          trust_remote_code=True).cuda().eval()
        self.vla.requires_grad_(False)
        self.unnorm_key = SUITE if SUITE in self.vla.norm_stats else SUITE + "_no_noops"
        self.stats = self.vla.norm_stats[self.unnorm_key]["action"]
        n = self.proc.image_processor.tvf_normalize_params
        self.mu = torch.tensor([*n[0]["mean"], *n[1]["mean"]], device="cuda").view(1, 6, 1, 1)
        self.sd = torch.tensor([*n[0]["std"], *n[1]["std"]], device="cuda").view(1, 6, 1, 1)
        self._ids = {}

    def ids(self, instr):
        if instr not in self._ids:
            x = self.proc.tokenizer(prompt(instr), return_tensors="pt").input_ids
            if x[0, -1] != 29871:  # same special empty token predict_action appends
                x = self.torch.cat([x, self.torch.tensor([[29871]])], 1)
            self._ids[instr] = x.cuda()
        return self._ids[instr]

    def to01(self, imgs):
        return self.torch.as_tensor(np.asarray(imgs), device="cuda").permute(0, 3, 1, 2).float() / 255

    def pixels(self, x01):  # differentiable PrismaticImageProcessor (resize is identity at 224; to_tensor; normalize x2)
        return ((x01.repeat(1, 2, 1, 1) - self.mu) / self.sd).to(self.torch.bfloat16)

    def act(self, imgs, instr, hidden=True):
        torch = self.torch
        with torch.no_grad():
            ids = self.ids(instr).expand(len(imgs), -1)
            out = self.vla(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=self.pixels(self.to01(imgs)),
                           use_cache=True, output_hidden_states=hidden)
            logits, past = [out.logits[:, -1]], out.past_key_values
            hid = torch.stack([out.hidden_states[l][:, -1] for l in LAYERS], 1).half().cpu().numpy() if hidden else None
            del out
            for _ in range(6):  # batched greedy decode, LLM only (= what the prismatic forward does for batch 1)
                o = self.vla.language_model(input_ids=logits[-1].argmax(-1, keepdim=True), past_key_values=past, use_cache=True)
                past = o.past_key_values
                logits.append(o.logits[:, -1])
            r = out_feats(torch.stack(logits, 1))
        r["hid"] = hid
        return r

    def train_patch(self, frames, steps=300, mb=2, accum=4, lr=0.03, seed=0, log=print):
        """Universal patch: under any other task's instruction, imitate the greedy action tokens the policy emits for
        LANG[TARGET] on the same (clean) frame. Open-loop teacher (no closed-loop refresh, unlike TRACO 2610.23566)."""
        torch = self.torch
        g = np.random.default_rng(seed)
        tgt = np.concatenate([self.act(frames[i:i + 16], LANG[TARGET], hidden=False)["toks"] for i in range(0, len(frames), 16)])
        tgt = torch.as_tensor(tgt, device="cuda")
        torch.manual_seed(seed)
        patch = torch.rand(3, PATCH, PATCH, device="cuda", requires_grad=True)
        opt = torch.optim.Adam([patch], lr=lr)
        src = [t for t in range(10) if t != TARGET]
        y, x = PATCH_YX
        hist = []
        for step in range(steps):
            tot = 0.0
            for _ in range(accum):
                idx = g.choice(len(frames), mb, replace=False)
                ids = self.ids(LANG[src[g.integers(len(src))]])
                ids = torch.cat([ids.expand(mb, -1), tgt[idx]], 1)
                im = self.to01(frames[idx]).clone()
                im[:, :, y:y + PATCH, x:x + PATCH] = patch.clamp(0, 1)
                lo = self.vla(input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=self.pixels(im)).logits[:, -8:-1]
                loss = torch.nn.functional.cross_entropy(lo.float().reshape(-1, lo.shape[-1]), tgt[idx].reshape(-1)) / accum
                loss.backward()
                tot += loss.item()
            opt.step()
            opt.zero_grad()
            with torch.no_grad():
                patch.clamp_(0, 1)
            hist.append(tot)
            if step % 25 == 0:
                log(f"[patch] step {step} loss {tot:.3f}")
        return (patch.detach().permute(1, 2, 0) * 255).round().byte().cpu().numpy(), hist


# ------------------------------------------------------------------------------------------------ LIBERO env workers
def libero_setup():
    """Write a LIBERO config (LIBERO's __init__ otherwise blocks on input()) and allow torch.load of init states."""
    import importlib.util
    root = os.path.join(importlib.util.find_spec("libero").submodule_search_locations[0], "libero")
    cfg = os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero_vk"))
    os.makedirs(cfg, exist_ok=True)
    with open(os.path.join(cfg, "config.yaml"), "w") as f:
        for k, v in dict(benchmark_root=root, bddl_files=f"{root}/bddl_files", init_states=f"{root}/init_files",
                         datasets=f"{root}/../datasets", assets=f"{root}/assets").items():
            f.write(f"{k}: {v}\n")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class LiberoEnv:
    """One LIBERO-Goal env; reused across episodes of the same scene task, reseeded per episode (deterministic)."""
    def __init__(self):
        libero_setup()
        from libero.libero import benchmark, get_libero_path
        import libero.libero.envs.bddl_utils as BU
        self.suite = benchmark.get_benchmark_dict()[SUITE]()
        assert [self.suite.get_task(i).name for i in range(10)] == TASK_NAMES
        self.bddl = [os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
                     for t in (self.suite.get_task(i) for i in range(10))]
        self.goals = [BU.robosuite_parse_problem(b)["goal_state"] for b in self.bddl]
        self.env, self.cur = None, None

    def goal_vec(self):
        e = self.env.env
        return np.array([all(e._eval_predicate(s) for s in g) for g in self.goals])

    def start(self, task, init):
        from libero.libero.envs import OffScreenRenderEnv
        if self.cur != task:
            if self.env is not None:
                self.env.close()
            self.env = OffScreenRenderEnv(bddl_file_name=self.bddl[task], camera_heights=256, camera_widths=256,
                                          camera_names=["agentview"])
            self.cur = task
        self.env.seed(0)
        self.env.reset()
        obs = self.env.set_init_state(self.suite.get_task_init_states(task)[init])
        for _ in range(WAIT):
            obs, _, _, _ = self.env.step(DUMMY)
        return prep(obs["agentview_image"]), self.goal_vec()

    def step(self, a):
        obs, _, _, _ = self.env.step(list(a))
        return prep(obs["agentview_image"]), self.goal_vec()


def _worker(conn):
    env = LiberoEnv()
    conn.send("ready")
    while True:
        cmd, arg = conn.recv()
        if cmd == "close":
            break
        try:
            conn.send(env.start(*arg) if cmd == "start" else env.step(arg))
        except Exception as e:  # surface sim errors in the parent instead of hanging
            conn.send(e)


class EnvPool:
    """n env worker processes (spawn, so CUDA in the parent is safe); send/recv per slot."""
    def __init__(self, n):
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.n, self.conns, self.procs = n, [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b,), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        assert all(c.recv() == "ready" for c in self.conns)

    def send(self, s, msg):
        self.conns[s].send(msg)

    def recv(self, s):
        r = self.conns[s].recv()
        if isinstance(r, Exception):
            raise r
        return r

    def close(self):
        for c, p in zip(self.conns, self.procs):
            c.send(("close", None))
            p.join(10)


# ------------------------------------------------------------------------------------------------ rollouts
def ends(e, goals):
    """Termination: task episodes on their own goal (standard eval); attack on own or TARGET goal; non-task on any goal."""
    if e["kind"] == "nontask":
        return goals.any()
    return goals[e["task"]] or (e["kind"] == "attack" and goals[TARGET])


def run_group(pol, pool, group, patch=None, max_steps=MAX_STEPS, log=print):
    """Run all episodes of one group (same instruction -> one batched prompt) on pool.n slots, refilling free slots."""
    eps, instr = group["episodes"], group["instr"]
    todo, active, recs = list(range(len(eps))), {}, [None] * len(eps)
    timing, t0 = [], time.time()
    while todo or active:
        ts = time.time()
        starts = []
        for s in range(pool.n):
            if s not in active and todo:
                i = todo.pop(0)
                pool.send(s, ("start", (eps[i]["task"], eps[i]["init"])))
                starts.append((s, i))
        for s, i in starts:
            img, goals = pool.recv(s)
            active[s] = dict(i=i, img=img, t=0, act=[], ent=[], var=[], hid=[], frames=[],
                             first=np.full(10, -1, np.int16), goals0=goals)
        slots = sorted(active)
        imgs = np.stack([active[s]["img"] for s in slots])
        if patch is not None:
            imgs = paste(imgs, patch, [eps[active[s]["i"]]["kind"] == "attack" for s in slots])
        need_h = [active[s]["t"] % CHUNK == 0 for s in slots]
        out = pol.act(imgs, instr, hidden=any(need_h))
        new = [k for k, s in enumerate(slots) if active[s]["t"] == 0]
        if new:  # counterfactual empty-instruction first action (2610.41609 gate)
            cf = pol.act(imgs[new], "", hidden=False)["norm"] if instr else out["norm"][new]
            for j, k in enumerate(new):
                active[slots[k]]["a0e"] = cf[j]
        for k, s in enumerate(slots):
            pool.send(s, ("step", env_action(out["norm"][k].copy(), pol.stats).tolist()))
        for k, s in enumerate(slots):
            st, e = active[s], eps[active[s]["i"]]
            img, goals = pool.recv(s)
            st["act"].append(out["norm"][k]); st["ent"].append(out["ent"][k]); st["var"].append(out["var"][k])
            if need_h[k]:
                st["hid"].append(out["hid"][k])
                if e["kind"] == "pilot":
                    st["frames"].append(imgs[k])
            st["t"] += 1
            new_g = goals & ~st["goals0"]  # goals already true at the start never count
            st["first"][new_g & (st["first"] < 0)] = st["t"]  # number of steps executed when the goal first held
            if ends(e, new_g) or st["t"] >= max_steps:
                recs[st["i"]] = dict(e, key=group["key"], instr=instr, T=st["t"], success=bool(st["first"][e["task"]] >= 0),
                                     first_goal=st["first"], goals_at_start=st["goals0"], a0e=st["a0e"],
                                     act=np.array(st["act"], np.float16), ent=np.array(st["ent"], np.float16),
                                     var=np.array(st["var"], np.float16), hid=np.array(st["hid"], np.float16),
                                     frames=np.array(st["frames"], np.uint8) if st["frames"] else None)
                del active[s]
            else:
                st["img"] = img
        timing.append((len(slots), time.time() - ts))  # wall time of one batched step incl. sim + env starts
    log(f"[{group['key']}] {len(eps)} eps in {time.time() - t0:.0f}s, success {np.mean([r['success'] for r in recs]):.2f}")
    return recs, timing


def plan(stage, reduced=False):
    """Episode plan. Init-state index ranges are disjoint per role: pilot 40-44 (tasks 0,2,4,6,8 + empty), clean 0-19, non-task 20-21, attack 23-28."""
    if stage == "pilot":
        G = [dict(key=f"pilot_t{t}", instr=LANG[t], episodes=[dict(kind="pilot", task=t, init=i) for i in range(40, 44)])
             for t in range(0, 10, 2)]
        return G + [dict(key="pilot_empty", instr="", episodes=[dict(kind="nontask", ntype="empty", task=t, init=44)
                                                                 for t in range(10)])]
    nc, na = (15, 4) if reduced else (20, 6)
    G = []
    for t in range(10):
        eps = [dict(kind="clean", task=t, init=i) for i in range(nc)]
        if t != TARGET:
            eps += [dict(kind="attack", task=t, init=23 + i) for i in range(na)]
        G.append(dict(key=f"task{t}", instr=LANG[t], episodes=eps))
    for nt, s in NONTASK.items():
        G.append(dict(key=f"nontask_{nt}", instr=s,
                      episodes=[dict(kind="nontask", ntype=nt, task=t, init=20 + i) for t in range(10) for i in range(2)]))
    return G
