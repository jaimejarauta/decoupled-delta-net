"""CPU checks on tiny random hybrids: the interventions are exact, local, and a no-op when W >= L."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hybrid_locality.evals import target_logprobs  # noqa: E402
from hybrid_locality.window import intervene, modules, staggered  # noqa: E402


def test_staggered_matches_bruteforce():
    x = torch.randn(2, 37, 3)
    fn = lambda z: z.cumsum(1)  # never forgets: output = sum of everything it was shown
    W, h = 8, 4
    y = staggered(fn, x, W, max_tokens=16)
    for t in range(37):
        s = 0 if t < h else (t // h - 1) * h
        assert t - s + 1 <= W
        torch.testing.assert_close(y[:, t], x[:, s : t + 1].sum(1))


def tiny(kind):
    from transformers import (FalconH1Config, GraniteMoeHybridConfig, Qwen3_5TextConfig, Zamba2Config,
                              AutoModelForCausalLM)
    common = dict(vocab_size=97, hidden_size=64, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2)
    if kind == "qwen3_5":
        cfg = Qwen3_5TextConfig(**common, num_hidden_layers=2, layer_types=["linear_attention", "full_attention"],
                                head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                                linear_key_head_dim=16, linear_value_head_dim=16)
    elif kind == "falcon_h1":
        cfg = FalconH1Config(**common, num_hidden_layers=2, mamba_d_ssm=64, mamba_n_heads=4, mamba_d_head=16,
                             mamba_d_state=8, mamba_n_groups=1, mamba_chunk_size=8, head_dim=16)
    elif kind == "zamba2":
        cfg = Zamba2Config(**common, num_hidden_layers=3, layers_block_type=["mamba", "hybrid", "mamba"],
                           mamba_d_state=8, mamba_headdim=16, n_mamba_heads=8, chunk_size=8,
                           num_mem_blocks=1, use_shared_attention_adapter=False, adapter_rank=8)
    elif kind == "granite":
        cfg = GraniteMoeHybridConfig(**common, num_hidden_layers=2, layer_types=["mamba", "attention"],
                                     mamba_d_state=8, mamba_d_head=16, mamba_n_heads=8, mamba_chunk_size=8,
                                     num_local_experts=0, shared_intermediate_size=64)
    cfg._attn_implementation = "sdpa"
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_config(cfg).eval()


KINDS = ["qwen3_5", "falcon_h1", "zamba2", "granite"]


@pytest.mark.parametrize("kind", KINDS)
@torch.no_grad()
def test_hybrid_interventions(kind):
    model = tiny(kind)
    lin_mods, att_mods = modules(model)
    assert lin_mods and att_mods
    L, W = 128, 8
    ids = torch.randint(0, 97, (1, L))
    ref = model(ids, use_cache=False).logits

    # our chunked logprobs reproduce the model's own logits (incl. Falcon/Granite logit scaling)
    lp = target_logprobs(model, ids, chunk=16)
    torch.testing.assert_close(lp, ref[0, :-1].float().log_softmax(-1).gather(-1, ids[0, 1:, None])[:, 0])

    # W >= L is a no-op, and the hooks really fire on every attention module
    with intervene(model, lin=L, attn=L):
        torch.testing.assert_close(model(ids, use_cache=False).logits, ref)
    assert all(getattr(m, "_hl_calls", 0) > 0 for m in att_mods)
    with intervene(model, lin=W):
        assert not torch.allclose(model(ids, use_cache=False).logits, ref)
    with intervene(model, attn=W):
        assert not torch.allclose(model(ids, use_cache=False).logits, ref)
    torch.testing.assert_close(model(ids, use_cache=False).logits, ref)  # hooks removed

    # locality: with every mixer windowed, token 0 cannot reach positions beyond the stacked receptive field
    ids2 = ids.clone()
    ids2[0, 0] = (ids[0, 0] + 1) % 97
    rf = len(lin_mods) * W + len(att_mods) * W  # loose upper bound (shared blocks are counted once per call site)
    rf = 2 * rf  # zamba2 reuses its shared block; keep the bound conservative
    assert rf < L
    with intervene(model, lin=W, attn=W):
        a, b = model(ids, use_cache=False).logits, model(ids2, use_cache=False).logits
    torch.testing.assert_close(a[:, rf:], b[:, rf:])
    assert not torch.allclose(model(ids, use_cache=False).logits[:, rf:], model(ids2, use_cache=False).logits[:, rf:])

    # trunc arm equals literally feeding the window
    lp_t = target_logprobs(model, ids, trunc=W)
    t, s = 41, (41 // (W // 2) - 1) * (W // 2)
    brute = model(ids[:, s : t + 1], use_cache=False).logits[0, -1].float().log_softmax(-1)[ids[0, t + 1]]
    torch.testing.assert_close(lp_t[t], brute)


def test_verdict_rule():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_kill import summarize, verdict

    def rows(model, acc):  # acc: arm -> fraction correct over 100 items
        out = [dict(model=model, kind="ruler", task="single", L=8192, arm="full", W=None, i=i, ok=True) for i in range(100)]
        for arm, a in acc.items():
            out += [dict(model=model, kind="ruler", task="single", L=8192, arm=arm, W=1024, i=i, ok=i < 100 * a) for i in range(100)]
        out += [dict(model=model, kind="ppl", L=8192, arm=arm, W=None if arm == "full" else 1024, i=0, key_nll=1.0, n_key=5, nll=1.0)
                for arm in ["full", *acc]]
        return out

    local = {"lin": 0.99, "attn": 0.1, "trunc": 0.0}
    assert verdict(summarize(sum((rows(f"m{k}", local) for k in range(3)), []), 1024))[0] == "H1"
    carries = {"lin": 0.5, "attn": 0.9, "trunc": 0.0}
    assert verdict(summarize(sum((rows(f"m{k}", carries) for k in range(3)), []), 1024))[0] == "H0"
    no_long = {"lin": 1.0, "attn": 1.0, "trunc": 0.95}  # tasks solvable locally -> cannot count
    assert verdict(summarize(sum((rows(f"m{k}", no_long) for k in range(3)), []), 1024))[0] == "INVALID"
