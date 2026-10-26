"""Decoupled Delta-Net Architecture: Linear recurrence with surprise-triggered error reservoir."""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ErrorReservoir:
    """Fixed-capacity circular reservoir caching high-surprise prediction residuals."""

    def __init__(self, capacity: int = 128, d_k: int = 32, d_v: int = 32, device: torch.device = torch.device("cpu"), dtype: torch.dtype = torch.float32):
        self.capacity = capacity
        self.d_k = d_k
        self.d_v = d_v
        self.device = device
        self.dtype = dtype
        self.k_buffer = torch.zeros(capacity, d_k, device=device, dtype=dtype)
        self.e_buffer = torch.zeros(capacity, d_v, device=device, dtype=dtype)
        self.ptr = 0
        self.count = 0

    def push(self, k: torch.Tensor, e: torch.Tensor):
        """Append a surprise key-error pair to the circular reservoir."""
        # k: (d_k,), e: (d_v,)
        self.k_buffer[self.ptr] = k.detach().to(self.dtype)
        self.e_buffer[self.ptr] = e.detach().to(self.dtype)
        self.ptr = (self.ptr + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def query(self, q: torch.Tensor) -> torch.Tensor:
        """Associative readout from reservoir: q in R^{d_k} -> readout in R^{d_v}."""
        if self.count == 0:
            return torch.zeros(self.d_v, device=self.device, dtype=self.dtype)
        valid_k = self.k_buffer[:self.count]  # (C, d_k)
        valid_e = self.e_buffer[:self.count]  # (C, d_v)
        scores = torch.matmul(valid_k, q.to(self.dtype)) / math.sqrt(self.d_k)  # (C,)
        weights = F.softmax(scores, dim=0)  # (C,)
        return torch.matmul(weights, valid_e)  # (d_v,)


class DecoupledDeltaCell(nn.Module):
    """Linear delta-rule recurrent cell coupled with a discrete surprise reservoir."""

    def __init__(
        self,
        d_model: int = 64,
        d_k: int = 32,
        d_v: int = 32,
        surprise_threshold: float = 0.5,
        reservoir_capacity: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.surprise_threshold = surprise_threshold
        self.reservoir_capacity = reservoir_capacity

        self.proj_q = nn.Linear(d_model, d_k, bias=False)
        self.proj_k = nn.Linear(d_model, d_k, bias=False)
        self.proj_v = nn.Linear(d_model, d_v, bias=False)
        self.proj_beta = nn.Linear(d_model, 1, bias=True)

        self.out_proj = nn.Linear(d_v, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Processes sequence (B, T, d_model) through decoupled delta recurrence.
        
        Accelerated with in-place fused state updates and batched reservoir processing,
        eliminating sequential batch loops and redundant inner tensor allocations.
        """
        b, t, d = x.shape
        device = x.device
        dtype = x.dtype

        q = self.proj_q(x)                            # (B, T, d_k)
        k = F.normalize(self.proj_k(x), p=2, dim=-1)  # (B, T, d_k)
        v = self.proj_v(x)                            # (B, T, d_v)
        beta = torch.sigmoid(self.proj_beta(x))        # (B, T, 1)

        y_out = torch.empty(b, t, self.d_v, device=device, dtype=dtype)
        surprises_logged = 0
        thresh_sq = self.surprise_threshold ** 2
        scale = 1.0 / math.sqrt(self.d_k)

        if b == 1:
            s_state = torch.zeros(self.d_k, self.d_v, device=device, dtype=dtype)
            reservoir = ErrorReservoir(capacity=self.reservoir_capacity, d_k=self.d_k, d_v=self.d_v, device=device, dtype=dtype)
            k_buf = reservoir.k_buffer
            e_buf = reservoir.e_buffer
            q_0 = q[0]
            k_0 = k[0]
            v_0 = v[0]
            beta_0 = beta[0]
            ptr = 0
            count = 0
            cap = self.reservoir_capacity

            for step in range(t):
                q_t = q_0[step]
                k_t = k_0[step]
                v_t = v_0[step]
                b_t = beta_0[step]

                v_pred = torch.matmul(k_t, s_state)
                err = v_t - v_pred
                s_state.addmm_(k_t.unsqueeze(1), (b_t * err).unsqueeze(0))
                y_fast = torch.matmul(q_t, s_state)

                err_sq = torch.dot(err, err).item() / self.d_v
                if err_sq > thresh_sq:
                    k_buf[ptr] = k_t
                    e_buf[ptr] = err
                    ptr = (ptr + 1) % cap
                    count = min(count + 1, cap)
                    surprises_logged += 1

                if count == 0:
                    y_out[0, step] = y_fast
                else:
                    scores = torch.matmul(k_buf[:count], q_t) * scale
                    weights = F.softmax(scores, dim=0)
                    y_slow = torch.matmul(weights, e_buf[:count])
                    y_out[0, step] = y_fast + y_slow
        else:
            s_state = torch.zeros(b, self.d_k, self.d_v, device=device, dtype=dtype)
            k_buf = torch.zeros(b, self.reservoir_capacity, self.d_k, device=device, dtype=dtype)
            e_buf = torch.zeros(b, self.reservoir_capacity, self.d_v, device=device, dtype=dtype)
            valid_mask = torch.zeros(b, self.reservoir_capacity, dtype=torch.bool, device=device)
            ptr = torch.zeros(b, dtype=torch.long, device=device)
            batch_indices = torch.arange(b, device=device)
            cap = self.reservoir_capacity

            for step in range(t):
                q_t = q[:, step]
                k_t = k[:, step]
                v_t = v[:, step]
                b_t = beta[:, step]

                v_pred = torch.bmm(k_t.unsqueeze(1), s_state).squeeze(1)
                err = v_t - v_pred
                s_state.baddbmm_(k_t.unsqueeze(2), (b_t * err).unsqueeze(1))
                y_fast = torch.bmm(q_t.unsqueeze(1), s_state).squeeze(1)

                err_sq = torch.mean(err * err, dim=-1)
                is_surprise = err_sq > thresh_sq
                surprises_logged += int(is_surprise.sum().item())

                curr_ptr = ptr
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
                y_out[:, step] = y_fast + y_slow

        out = self.out_proj(y_out)
        telemetry = {
            "surprise_rate": surprises_logged / (b * t + 1e-7),
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
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cell = DecoupledDeltaCell(
            d_model=d_model, d_k=d_k, d_v=d_v, surprise_threshold=surprise_threshold
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
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            DecoupledDeltaBlock(
                d_model=d_model, d_k=d_k, d_v=d_v, d_ff=d_ff, surprise_threshold=surprise_threshold
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
