"""Decoupled Delta-Net Architecture: Linear recurrence with surprise-triggered error reservoir."""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoupledDeltaCell(nn.Module):
    """Linear delta-rule recurrent cell coupled with a discrete surprise reservoir."""

    def __init__(
        self,
        d_model: int = 64,
        d_k: int = 32,
        d_v: int = 32,
        surprise_threshold: float = 0.5,
        reservoir_capacity: int = 64,
        relative: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.surprise_threshold = surprise_threshold
        self.reservoir_capacity = reservoir_capacity
        # relative: surprise if err^2 > threshold * running mean err^2 (scale-free); else absolute err RMS > threshold
        self.relative = relative

        self.proj_q = nn.Linear(d_model, d_k, bias=False)
        self.proj_k = nn.Linear(d_model, d_k, bias=False)
        self.proj_v = nn.Linear(d_model, d_v, bias=False)
        self.proj_beta = nn.Linear(d_model, 1, bias=True)

        self.out_proj = nn.Linear(d_v, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Processes sequence (B, T, d_model) through the delta recurrence plus the surprise reservoir."""
        b, t, d = x.shape
        device = x.device
        dtype = x.dtype

        q = self.proj_q(x)                            # (B, T, d_k)
        k = F.normalize(self.proj_k(x), p=2, dim=-1)  # (B, T, d_k)
        v = self.proj_v(x)                            # (B, T, d_v)
        beta = torch.sigmoid(self.proj_beta(x))        # (B, T, 1)

        surprises_logged = torch.zeros((), device=device)
        thresh_sq = self.surprise_threshold ** 2
        scale = 1.0 / math.sqrt(self.d_k)

        s_state = torch.zeros(b, self.d_k, self.d_v, device=device, dtype=dtype)
        k_buf = torch.zeros(b, self.reservoir_capacity, self.d_k, device=device, dtype=dtype)
        e_buf = torch.zeros(b, self.reservoir_capacity, self.d_v, device=device, dtype=dtype)
        valid_mask = torch.zeros(b, self.reservoir_capacity, dtype=torch.bool, device=device)
        ptr = torch.zeros(b, dtype=torch.long, device=device)
        batch_indices = torch.arange(b, device=device)
        cap = self.reservoir_capacity
        y_steps = []

        for step in range(t):
            q_t = q[:, step]
            k_t = k[:, step]
            v_t = v[:, step]
            b_t = beta[:, step]

            v_pred = torch.bmm(k_t.unsqueeze(1), s_state).squeeze(1)
            err = v_t - v_pred
            if torch.is_grad_enabled():
                s_state = torch.baddbmm(s_state, k_t.unsqueeze(2), (b_t * err).unsqueeze(1))
            else:
                s_state.baddbmm_(k_t.unsqueeze(2), (b_t * err).unsqueeze(1))
            y_fast = torch.bmm(q_t.unsqueeze(1), s_state).squeeze(1)

            err_sq = torch.mean(err * err, dim=-1).detach()
            if self.relative:
                # running mean of err^2 per row, seeded at step 0 (no surprise on the first token)
                ema_sq = err_sq if step == 0 else 0.95 * ema_sq + 0.05 * err_sq
                is_surprise = (err_sq > self.surprise_threshold * ema_sq) & (step > 0)
            else:
                is_surprise = err_sq > thresh_sq
            surprises_logged = surprises_logged + is_surprise.sum()

            curr_ptr = ptr
            if torch.is_grad_enabled():
                ptr_mask = (torch.arange(cap, device=device).unsqueeze(0) == curr_ptr.unsqueeze(1)).unsqueeze(-1)
                update_mask = ptr_mask & is_surprise.unsqueeze(1).unsqueeze(2)
                k_buf = torch.where(update_mask, k_t.unsqueeze(1), k_buf)
                e_buf = torch.where(update_mask, err.unsqueeze(1), e_buf)
                valid_mask = torch.where(ptr_mask.squeeze(-1) & is_surprise.unsqueeze(1), True, valid_mask)
            else:
                k_buf[batch_indices, curr_ptr] = torch.where(is_surprise.unsqueeze(-1), k_t, k_buf[batch_indices, curr_ptr])
                e_buf[batch_indices, curr_ptr] = torch.where(is_surprise.unsqueeze(-1), err, e_buf[batch_indices, curr_ptr])
                valid_mask[batch_indices, curr_ptr] = is_surprise | valid_mask[batch_indices, curr_ptr]
            ptr = (curr_ptr + is_surprise.long()) % cap

            scores = torch.bmm(k_buf, q_t.unsqueeze(-1)).squeeze(-1) * scale
            scores = scores.masked_fill(~valid_mask, -1e4)
            weights = F.softmax(scores, dim=-1)
            y_slow = torch.bmm(weights.unsqueeze(1), e_buf).squeeze(1)
            has_valid = valid_mask.any(dim=-1, keepdim=True)
            y_slow = torch.where(has_valid, y_slow, torch.zeros_like(y_slow))
            y_steps.append(y_fast + y_slow)

        y_out = torch.stack(y_steps, dim=1)

        out = self.out_proj(y_out)
        telemetry = {
            "surprise_rate": surprises_logged.item() / (b * t + 1e-7),
        }
        return out, telemetry


class DecoupledDeltaBlock(nn.Module):
    """Full Decoupled Delta-Net layer with pre-norm, cell, and MLP."""

    def __init__(
        self,
        d_model: int = 64,
        d_k: int = 32,
        d_v: int = 32,
        d_ff: int = 128,
        surprise_threshold: float = 0.5,
        dropout: float = 0.05,
        relative: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cell = DecoupledDeltaCell(
            d_model=d_model, d_k=d_k, d_v=d_v, surprise_threshold=surprise_threshold, relative=relative
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        h, telem = self.cell(self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, telem


class DecoupledDeltaModel(nn.Module):
    """Decoupled Delta-Net language and sequence model."""

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 64,
        d_k: int = 32,
        d_v: int = 32,
        d_ff: int = 128,
        num_layers: int = 2,
        surprise_threshold: float = 0.5,
        relative: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoupledDeltaBlock(
                d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff, surprise_threshold=surprise_threshold,
                relative=relative,
            )
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        x = self.embed(input_ids)
        total_telem = {}
        for idx, layer in enumerate(self.layers):
            x, telem = layer(x)
            total_telem[f"layer_{idx}_surprise_rate"] = telem["surprise_rate"]
        logits = self.head(x)
        return logits, total_telem
