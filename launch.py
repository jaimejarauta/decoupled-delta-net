"""Build a batched run config (arms x seeds) and launch grok.py in the background."""
import json, subprocess, sys

PY = "/Users/jaimejarauta/Desktop/Others/Project/robotics/moonshot/.venv-nova/bin/python"


def launch(name, task, frac, arms, seeds, steps, device="mps"):
    """arms: list of (wd, gf_lambda, ctrl_cfg_or_None)."""
    cfg = dict(task=task, frac=frac, steps=steps, tag=name, out=f"runs/{name}.json", device=device,
               seeds=[s for _ in arms for s in seeds], wds=[a[0] for a in arms for _ in seeds],
               gf_lambdas=[a[1] for a in arms for _ in seeds], ctrl_cfgs=[a[2] for a in arms for _ in seeds])
    log = open(f"runs/{name}.log", "w")
    subprocess.Popen([PY, "grok.py", json.dumps(cfg)], stdout=log, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    exec(sys.argv[1])
