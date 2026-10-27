"""Long-run plumbing: resumable checkpoints, provenance manifest, wall-clock budget guard.
Identical copy in each repo (the repos are independent packages); see research/COLAB_LONG_RUNS_2026-10-26.md.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Optional

import torch


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def file_digest(path: str, chunk: int = 1 << 26) -> str:
    """sha256 of size + first and last 64 MB: cheap, and changes when the file changes."""
    h = hashlib.sha256(str(os.path.getsize(path)).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if os.path.getsize(path) > chunk:
            f.seek(-chunk, os.SEEK_END)
            h.update(f.read(chunk))
    return h.hexdigest()[:16]


class Run:
    """One resumable run living in out_dir: checkpoint.pt, manifest.json."""

    def __init__(self, out_dir: str, config: dict, budget_hours: Optional[float] = None):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.config = config
        self.budget_hours = budget_hours
        self.ckpt_path = os.path.join(out_dir, "checkpoint.pt")
        self.t_start = time.time()
        self.elapsed_before = 0.0  # wall-clock spent in earlier sessions (restored on resume)
        self.last_save = time.time()
        self.data_digests = {}

    @property
    def elapsed(self) -> float:
        return self.elapsed_before + time.time() - self.t_start

    def resume(self) -> Optional[dict]:
        """Latest checkpoint state, or None for a fresh run."""
        if not os.path.exists(self.ckpt_path):
            return None
        state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
        self.elapsed_before = state.get("_elapsed", 0.0)
        print(f"[run] resumed from {self.ckpt_path} ({self.elapsed_before / 3600:.2f} h already spent)")
        return state

    def save(self, state: dict) -> None:
        """Atomic write, so a disconnect mid-save never leaves a corrupt checkpoint."""
        state = dict(state, _elapsed=self.elapsed)
        tmp = self.ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)
        self.last_save = time.time()

    def due(self, every_sec: float = 900.0) -> bool:
        return time.time() - self.last_save >= every_sec

    def check_budget(self, done: float, total: float) -> None:
        """Stop if the projected wall-clock exceeds 2x the budget line of the plan."""
        if not self.budget_hours or done <= 0:
            return
        projected = self.elapsed / done * total / 3600.0
        if projected > 2.0 * self.budget_hours:
            raise SystemExit(f"[run] projected {projected:.1f} h > 2 x budget {self.budget_hours:.1f} h; stopping")

    def log_throughput(self, done: float, total: float, unit: str) -> None:
        rate = done / max(self.elapsed, 1e-6)
        projected = total / max(rate, 1e-9) / 3600.0
        print(f"[run] {rate:,.1f} {unit}/s | projected {projected:.2f} h total (budget {self.budget_hours} h)")

    def write_manifest(self, results: dict) -> str:
        manifest = {
            "git_commit": git_hash(),
            "command": " ".join(sys.argv),
            "config": self.config,
            "data_digests": self.data_digests,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu/mps",
            "elapsed_hours": self.elapsed / 3600.0,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": results,
        }
        path = os.path.join(self.out_dir, "manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        print(f"[run] manifest: {path}")
        return path
