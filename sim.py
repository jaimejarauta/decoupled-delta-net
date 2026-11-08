"""LIBERO (hf-libero, via LeRobot's LiberoEnv) + LeRobot policy adapter.

- ReachEnv: exact snapshot/restore (MuJoCo state + warm-start + ctrl + gripper + OSC controller attributes; the
  sim state alone is NOT enough across env instances, see test_reach.py), per-step unsafe features.
- Worker pool (spawn; one env per process): particles move between processes as snapshots, so any worker can
  continue any particle.
- Policy: predict_action_chunk on a batch, first K actions executed (policy is stateless between decisions, so a
  particle = (sim snapshot, step t, running-max score M, flags) is a Markov state at decision points).
- Runner.start / Runner.advance are the `start` / `advance` callables of reach.splitting."""
import os, sys, copy, itertools, importlib.util, multiprocessing as mp
import numpy as np

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

SUITE, T_MAX, K, RES, N_INIT = "libero_goal", 300, 20, 256, 50
DUMMY = np.array([0, 0, 0, 0, 0, 0, -1], np.float32)


def _libero_config():
    """hf-libero asks input() on first import if no config exists -> write the default one first."""
    d = os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
    f = os.path.join(d, "config.yaml")
    if not os.path.exists(f):
        root = os.path.join(list(importlib.util.find_spec("libero").submodule_search_locations)[0], "libero")
        os.makedirs(d, exist_ok=True)
        paths = dict(benchmark_root=root, bddl_files=f"{root}/bddl_files", init_states=f"{root}/init_files",
                     datasets=f"{root}/../datasets", assets=f"{root}/assets")
        open(f, "w").write("".join(f"{k}: {v}\n" for k, v in paths.items()))


_libero_config()
import mujoco  # noqa: E402
from libero.libero import benchmark  # noqa: E402
from lerobot.sim.integrations.libero import LiberoEnv, canonical_observation  # noqa: E402
import reach  # noqa: E402


def suite_obj(name=SUITE):
    return benchmark.get_benchmark_dict()[name]()


class ReachEnv(LiberoEnv):
    def prepare(self):
        self.reset()                                   # builds the env (hard reset once) at init state 0
        e = self._env.env
        self.names = list(e.objects_dict)
        self.ids = np.array([e.obj_body_id[n] for n in self.names])
        ooi = list(e.obj_of_interest)
        self.distractor = np.array([not any(o == n or o.startswith(n + "_") for o in ooi) for n in self.names])
        m = e.sim.model._model
        rb = {b for b in range(m.nbody) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(("robot0_", "gripper0_"))}
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in rb}
        self.z0 = self.snapshot()                      # canonical controller/gripper state after a real reset
        return self

    # -- state
    def snapshot(self):
        e = self._env.env
        dd, r = e.sim.data._data, e.robots[0]
        return dict(state=self._env.get_sim_state().copy(), warm=dd.qacc_warmstart.copy(), ctrl=dd.ctrl.copy(),
                    grip=np.array(r.gripper.current_action, float), t_env=int(e.timestep),
                    ctl={k: copy.deepcopy(v) for k, v in vars(r.controller).items() if isinstance(v, (np.ndarray, float, int, bool))})

    def restore(self, z):
        raw = self._env.set_init_state(z["state"])
        e = self._env.env
        dd, r = e.sim.data._data, e.robots[0]
        e.timestep, e.done = z["t_env"], False
        dd.qacc_warmstart[:], dd.ctrl[:] = z["warm"], z["ctrl"]
        r.gripper.current_action = z["grip"].copy()
        for k, v in z["ctl"].items():
            setattr(r.controller, k, copy.deepcopy(v))
        return raw

    # -- unsafe features of the current sim state
    def features(self, raw=None):
        e = self._env.env
        m, dd = e.sim.model._model, e.sim.data._data
        f6, force = np.zeros(6), 0.0
        for i in range(dd.ncon):
            c = dd.contact[i]
            if c.geom1 in self.robot_geoms or c.geom2 in self.robot_geoms:
                mujoco.mj_contactForce(m, dd, i, f6)
                force = max(force, abs(f6[0]))
        return dict(pos=dd.xpos[self.ids].copy(), zax=dd.xmat[self.ids].reshape(-1, 3, 3)[:, :, 2].copy(),
                    eef=dd.site_xpos[e.robots[0].eef_site_id].copy(), force=force)   # sim state, not the (lagged) observable

    def obs(self, raw):
        return canonical_observation(self._format_raw_obs(raw))

    # -- particle API
    def start(self, init_id, active):
        z = dict(self.z0, state=self._init_states[init_id % len(self._init_states)], t_env=0)
        raw = self.restore(z)
        for _ in range(self.num_steps_wait):           # LIBERO settle steps (not counted as policy steps)
            raw, _, _, _ = self._env.step(DUMMY)
        f = self.features(raw)
        ref = dict(pos=f["pos"], zax=f["zax"], eef=f["eef"])
        return dict(z=self.snapshot(), obs=self.obs(raw), t=0, M=0.0, ref=ref, init=int(init_id), active=list(active),
                    flags={k: False for k in reach.PREDICATES} | {f"mild_{k}": False for k in reach.MILD},
                    done=False, success=False, steps=0)

    def segment(self, p, actions, halt):
        raw = self.restore(p["z"])
        t, M, flags, steps, success = p["t"], p["M"], dict(p["flags"]), 0, False
        for a in actions:
            raw, _, _, _ = self._env.step(np.asarray(a, np.float64))
            t, steps = t + 1, steps + 1
            success = bool(self._env.check_success())
            m, prox, mild = reach.margins(self.features(raw), p["ref"], self.distractor)
            M = max(M, reach.score(m, prox, p["active"]))
            for k in p["active"]:
                flags[k] |= m[k] >= 1
            for k, v in mild.items():
                flags[f"mild_{k}"] |= v
            if success or t >= T_MAX or M >= halt:
                break
        return dict(p, z=self.snapshot(), obs=self.obs(raw), t=t, M=M, flags=flags, steps=p["steps"] + steps,
                    success=success, done=success or t >= T_MAX)


# ---------------------------------------------------------------- worker processes
_E = None


def _w_init(suite_name, task_id, res):
    global _E
    _E = ReachEnv(suite_obj(suite_name), task_id, suite_name, obs_type="pixels_agent_pos",
                  observation_width=res, observation_height=res).prepare()


def _w_start(a):
    return _E.start(*a)


def _w_segment(a):
    return _E.segment(*a)


def _w_info(_):
    return dict(names=_E.names, distractor=_E.distractor.tolist(), ooi=list(_E._env.env.obj_of_interest))


class Runner:
    def __init__(self, policy, task_id, workers=None, suite_name=SUITE, k=K, res=RES, active=reach.PREDICATES, seed=0):
        self.policy, self.k, self.active, self.steps = policy, k, list(active), 0
        self.task = suite_obj(suite_name).get_task(task_id).language
        self.workers = workers or min(16, os.cpu_count() or 2)
        self.pool = mp.get_context("spawn").Pool(self.workers, initializer=_w_init, initargs=(suite_name, task_id, res))
        self.seeds = itertools.count(seed * 1_000_003)
        self.info = self.pool.map(_w_info, [0])[0]

    def start(self, init_ids):
        return self.pool.map(_w_start, [(int(i), self.active) for i in init_ids])

    def start_random(self, n, rng):
        return self.start(rng.integers(N_INIT, size=n))

    def advance(self, parts, stop, halt=1.0):
        """Run particles until M >= stop or episode end; halt mid-chunk once M >= halt (unsafe = absorbing)."""
        fin, act = [], []
        for p in parts:
            (fin if p["M"] >= stop or p["done"] else act).append(dict(p, steps=0))
        while act:
            A = self.policy.act([p["obs"] for p in act], self.task, self.k, next(self.seeds))
            out = self.pool.map(_w_segment, [(dict(p, obs=None), a, halt) for p, a in zip(act, A)], chunksize=1)
            self.steps += sum(o["steps"] - p["steps"] for o, p in zip(out, act))
            act = []
            for o in out:
                (fin if o["M"] >= stop or o["done"] else act).append(o)
        return fin

    def episodes(self, init_ids):
        """Standard i.i.d. evaluation: full episodes (no halt at unsafe), as LIBERO evaluates success."""
        return self.advance(self.start(init_ids), float("inf"), float("inf"))

    def close(self):
        self.pool.terminate()
        self.pool.join()


# ---------------------------------------------------------------- policies
class Policy:
    """LeRobot policy, batched; stochastic as deployed (flow-matching noise ~ N(0, I) drawn per decision)."""

    def __init__(self, repo, device="cuda", bs=32):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        cfg = PreTrainedConfig.from_pretrained(repo)
        cfg.device = device
        cfg.pretrained_path = repo
        self.policy = get_policy_class(cfg.type).from_pretrained(repo, config=cfg).to(device).eval()
        self.policy.reset()
        img = "observation.images."
        rename = {f"{img}image": f"{img}camera1", f"{img}image2": f"{img}camera2"} if f"{img}camera1" in cfg.input_features else {}
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=repo, preprocessor_overrides={
            "device_processor": {"device": device}, "rename_observations_processor": {"rename_map": rename}})
        self.torch, self.bs, self.repo = torch, bs, repo

    def act(self, obs, task, k, seed):
        from lerobot.envs.utils import prepare_canonical_observation
        torch = self.torch
        torch.manual_seed(seed)
        out = []
        with torch.inference_mode():
            for i in range(0, len(obs), self.bs):
                ob = obs[i:i + self.bs]
                b = prepare_canonical_observation({key: np.stack([o[key] for o in ob]) for key in ob[0]})
                b["task"] = [task] * len(ob)
                a = self.policy.predict_action_chunk(self.pre(b))[:, :k]
                n, kk, d = a.shape
                a = self.post(a.reshape(n * kk, d))
                out.append(a.reshape(n, kk, -1)[..., :7].float().cpu().numpy())
        return np.concatenate(out)

    def free(self):
        del self.policy
        self.torch.cuda.empty_cache()


class RandomPolicy:
    """CPU stand-in for tests: uniform random deltas, gripper random sign."""

    def free(self):
        pass

    def __init__(self, scale=1.0):
        self.scale = scale

    def act(self, obs, task, k, seed):
        a = np.random.default_rng(seed).uniform(-1, 1, (len(obs), k, 7)) * self.scale
        a[..., 6] = np.sign(a[..., 6] + 1e-9)
        return a.astype(np.float32)
