"""Live autoregressive generation with step-boundary online checkpointing and
strictly budget-matched rollback (Arm 2 of C1 Kill Experiment).

Crucial anti-leakage / fair-accounting invariant:
Every discarded dead-end token, chunked-replay token, and negative-lemma prefill token
is deducted from `max_total_new_tokens`, so rollback arms never exceed the compute budget
of uninterrupted standard Chain-of-Thought decoding.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .cache_state import (
    capture_recurrent_snapshot,
    prefill_tokens,
    rollback_attention_only_kv,
    rollback_synchronized,
)
from .revolve import OnlineCheckpointPool
from .tasks import detect_backtrack_trigger


@dataclass
class LiveDecodeResult:
    text: str
    surviving_tokens: int
    total_compute_tokens: int
    replay_tokens: int
    backtracks_triggered: int


@torch.no_grad()
def live_decode_with_rollback(
    model,
    tok,
    prompt_ids: list[int],
    arm: str = "sync_binomial",
    max_total_compute_tokens: int = 1024,
    pool_cap: int = 8,
    min_dead_end_tokens: int = 128,
    max_backtracks: int = 2,
    temperature: float = 0.0,
    seed: int = 0,
    device: str = "cpu",
) -> LiveDecodeResult:
    """Autoregressively decode under a hard `max_total_compute_tokens` compute cap.

    `arm` in {'uninterrupted', 'attn_kv_only', 'prompt_restart', 'full_replay', 'ring_fifo', 'sync_binomial'}.
    """
    policy_map = {"full_replay": "zero", "ring_fifo": "ring_fifo", "sync_binomial": "min_span"}
    pool = OnlineCheckpointPool(capacity=pool_cap, policy=policy_map.get(arm, "zero"))

    rng = torch.Generator(device=device).manual_seed(seed)
    seq: list[int] = list(prompt_ids)
    prompt_len = len(prompt_ids)
    pkv, last_logits = prefill_tokens(model, torch.tensor([seq], dtype=torch.long, device=device))
    pool.push(capture_recurrent_snapshot(pkv, prompt_len))

    step_positions: list[int] = [prompt_len]
    compute_used = 0
    replay_total = 0
    backtracks = 0
    step_buf_ids: list[int] = []

    def _sample(logits_1d: torch.Tensor) -> int:
        if temperature <= 1e-6:
            return int(logits_1d.argmax(-1).item())
        probs = torch.softmax(logits_1d.float() / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1, generator=rng).item())

    while compute_used < max_total_compute_tokens:
        nxt = _sample(last_logits[0, -1])
        seq.append(nxt)
        step_buf_ids.append(nxt)
        compute_used += 1

        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break

        step_txt = tok.decode(step_buf_ids, skip_special_tokens=True)
        hit_boundary = step_txt.endswith("\n\n")

        if (
            arm != "uninterrupted"
            and backtracks < max_backtracks
            and detect_backtrack_trigger(step_txt)
            and (len(seq) - step_positions[0]) >= min_dead_end_tokens
        ):
            # Target step `t_a`: earliest retained step boundary before the current flawed segment
            candidates = [p for p in step_positions if len(seq) - p >= min_dead_end_tokens]
            t_a = candidates[-1] if candidates else prompt_len
            dead_snippet = tok.decode(seq[t_a : t_a + 16], skip_special_tokens=True).strip().replace("\n", " ")[:48]
            lemma_ids = tok(f"[Discarded contradicted branch: {dead_snippet}]\n\n", add_special_tokens=False).input_ids[:24]

            if arm == "prompt_restart":
                seq = list(prompt_ids) + lemma_ids
                cost = len(seq)
                if compute_used + cost >= max_total_compute_tokens:
                    break
                pkv, last_logits = prefill_tokens(model, torch.tensor([seq], dtype=torch.long, device=device))
                compute_used += cost
                replay_total += len(prompt_ids)
                pool.clear()
                pool.push(capture_recurrent_snapshot(pkv, len(seq)))
                step_positions = [len(seq)]
            elif arm == "attn_kv_only":
                if compute_used + len(lemma_ids) >= max_total_compute_tokens:
                    break
                rollback_attention_only_kv(pkv, t_a)
                seq = seq[:t_a] + lemma_ids
                pkv, last_logits = prefill_tokens(
                    model, torch.tensor([lemma_ids], dtype=torch.long, device=device), past_key_values=pkv
                )
                compute_used += len(lemma_ids)
                step_positions = [p for p in step_positions if p <= t_a] + [len(seq)]
            else:
                prefix_t = torch.tensor([seq[:t_a]], dtype=torch.long, device=device)
                rb = rollback_synchronized(model, pkv, prefix_t, pool, target_pos=t_a)
                cost = rb.replay_tokens + len(lemma_ids)
                if compute_used + cost >= max_total_compute_tokens:
                    break
                seq = seq[:t_a] + lemma_ids
                pkv, last_logits = prefill_tokens(
                    model, torch.tensor([lemma_ids], dtype=torch.long, device=device), past_key_values=pkv
                )
                compute_used += cost
                replay_total += rb.replay_tokens
                pool.push(capture_recurrent_snapshot(pkv, len(seq)))
                step_positions = [p for p in step_positions if p <= t_a] + [len(seq)]

            backtracks += 1
            step_buf_ids.clear()
            continue

        # Advance 1 step normally
        inp = torch.tensor([[nxt]], dtype=torch.long, device=device)
        out = model(input_ids=inp, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        last_logits = out.logits[:, -1:, :]

        if hit_boundary:
            cur_pos = len(seq)
            pool.push(capture_recurrent_snapshot(pkv, cur_pos))
            step_positions.append(cur_pos)
            step_buf_ids.clear()

    gen_text = tok.decode(seq[prompt_len:], skip_special_tokens=True)
    return LiveDecodeResult(
        text=gen_text,
        surviving_tokens=len(seq) - prompt_len,
        total_compute_tokens=prompt_len + compute_used,
        replay_tokens=replay_total,
        backtracks_triggered=backtracks,
    )

