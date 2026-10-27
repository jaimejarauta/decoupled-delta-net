"""Tests for the long-run plumbing (resume, atomic save, budget guard, manifest)."""

import json
import os

import pytest
import torch

from decoupled_delta.runtime import Run


def test_resume_roundtrip(tmp_path):
    run = Run(str(tmp_path), {"a": 1})
    assert run.resume() is None
    run.save({"step": 7, "w": torch.ones(2)})
    again = Run(str(tmp_path), {"a": 1})
    state = again.resume()
    assert state["step"] == 7 and torch.equal(state["w"], torch.ones(2))
    assert not os.path.exists(run.ckpt_path + ".tmp")


def test_budget_guard_stops_overrun(tmp_path):
    run = Run(str(tmp_path), {}, budget_hours=1.0)
    run.elapsed_before = 3600.0  # 1 h spent for 10% of the work -> 10 h projected
    with pytest.raises(SystemExit):
        run.check_budget(done=1, total=10)
    run.check_budget(done=9, total=10)  # 1.1 h projected: fine


def test_manifest_fields(tmp_path):
    run = Run(str(tmp_path), {"lr": 0.1})
    path = run.write_manifest({"metric": 1.0})
    m = json.load(open(path))
    assert m["config"]["lr"] == 0.1 and m["results"]["metric"] == 1.0 and "git_commit" in m
