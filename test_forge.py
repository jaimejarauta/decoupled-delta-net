import json
import mlx.core as mx
import pytest
import forge


@pytest.mark.parametrize("t", "ABC")
def test_baselines_pass_own_check(t):
    for f in (forge.builtin(t), forge.compiled(t)):
        assert forge.correctness(t, f, 16)[0]


@pytest.mark.parametrize("t", "ABC")
def test_hand_kernels_correct(t):
    assert forge.correctness(t, forge.build(t, json.load(open(f"hand/{t}.json"))), 16)[0]


def test_harness_rejects_cheats():
    # a kernel that only processes the first element, and one that writes zeros
    lazy = {"threadgroup": [1, 1, 1], "grid": ["1", "1", "1"],
            "source": "float gv = float(g[0]); out[0] = half(gv / (1.0f + metal::exp(-gv)) * float(u[0]));"}
    zero = {"threadgroup": [256, 1, 1], "grid": ["rows*I", "1", "1"],
            "source": "uint i = thread_position_in_grid.x; if (i < (uint)(rows*I)) out[i] = 0;"}
    for spec in (lazy, zero):
        assert not forge.correctness("B", forge.build("B", spec), 16)[0]


def test_timing_evaluates_all_outputs():
    # non-chained C shape: every call in the sample must be evaluated, not just the last
    t, shape = "C", (4096, 14336)
    inp = forge.make_inputs(t, shape, 0)
    mx.eval(inp)
    f = forge.builtin(t)
    forge.sample(t, f, inp, 4)
    t1 = min(forge.sample(t, f, inp, 1)[0] for _ in range(5))
    t8 = min(forge.sample(t, f, inp, 8)[0] for _ in range(5))
    assert t8 > 0.5 * t1  # per-call time must not collapse when K grows
