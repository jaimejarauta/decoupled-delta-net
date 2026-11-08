"""Parallel LIBERO envs, one spawned process per episode slot (no torch import here, keeps workers light).
Each reset takes an explicit init-state id, so every arm/alpha cell sees the same initial states (paired design).
Uses lerobot's own LiberoEnv + canonical_observation, i.e. the exact obs format lerobot-eval feeds the policy."""
import os
import multiprocessing as mp
import traceback


def _worker(conn, suite, task_id, size):
    try:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark
        from lerobot.sim.integrations.libero import LiberoEnv, canonical_observation
        env = LiberoEnv(benchmark.get_benchmark_dict()[suite](), task_id, suite, obs_type="pixels_agent_pos",
                        observation_height=size, observation_width=size)
        conn.send("ok")
    except Exception:
        conn.send(RuntimeError(traceback.format_exc()))
        return
    while True:
        cmd, arg = conn.recv()
        try:
            if cmd == "reset":
                env.init_state_id = arg
                obs, _ = env.reset(seed=arg)
                conn.send(canonical_observation(obs))
            elif cmd == "step":
                obs, _, term, _, info = env.step(arg)
                conn.send((canonical_observation(obs), bool(info["is_success"]), bool(term)))
            elif cmd == "close":
                env.close()
                conn.send("ok")
                return
        except Exception:
            conn.send(RuntimeError(traceback.format_exc()))


class Pool:
    def __init__(self, suite, task_id, n, size=256):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, suite, task_id, size), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)
        for c in self.conns:
            self._get(c)

    @staticmethod
    def _get(c):
        m = c.recv()
        if isinstance(m, Exception):
            raise m
        return m

    def reset(self, init_ids):
        assert len(init_ids) <= len(self.conns)
        for c, i in zip(self.conns, init_ids):
            c.send(("reset", int(i)))
        return [self._get(c) for c in self.conns[:len(init_ids)]]

    def step(self, actions):
        """actions: {slot: np.ndarray[7]} for running slots -> {slot: (obs, success, terminated)}."""
        for i, a in actions.items():
            self.conns[i].send(("step", a))
        return {i: self._get(self.conns[i]) for i in actions}

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None))
                c.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.kill()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
