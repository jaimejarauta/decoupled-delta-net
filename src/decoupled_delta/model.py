"""Decoupled Delta-Net Architecture: Linear recurrence with surprise-triggered error reservoir."""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ErrorReservoir:
    """Fixed-capacity circular reservoir caching high-surprise prediction residuals."""

    def __init__(self, capacity: int = 128, d_k: int = 32, d_v: int = 32, device: torch.device = torch.device("cpu")):
        self.capacity = capacity
        self.d_k = d_k
        self.d_v = d_v
        self.device = device
        self.k_buffer = torch.zeros(capacity, d_k, device=device)
        self.e_buffer = torch.zeros(capacity, d_v, device=device)
        self.ptr = 0
        self.count = 0

    def push(self, k: torch.Tensor, e: torch.Tensor):
        """Append a surprise key-error pair to the circular reservoir."""
        # k: (d_k,), e: (d_v,)
        self.k_buffer[self.ptr] = k.detach()
        self.e_buffer[self.ptr] = e.detach()
        self.ptr = (self.ptr + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def query(self, q: torch.Tensor) -> torch.Tensor:
        """Associative readout from reservoir: q in R^{d_k} -> readout in R^{d_v}."""
        if self.count == 0:
            return torch.zeros(self.d_v, device=self.device)
        valid_k = self.k_buffer[:self.count]  # (C, d_k)
        valid_e = self.e_buffer[:self.count]  # (C, d_v)
        scores = torch.matmul(valid_k, q) / math.sqrt(self.d_k)  # (C,)
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
        """Processes sequence (B, T, d_model) through decoupled delta recurrence."""
        b, t, d = x.shape
        device = x.device

        q = self.proj_q(x)  # (B, T, d_k)
        k = F.normalize(self.proj_k(x), p=2, dim=-1)  # (B, T, d_k)
        v = self.proj_v(x)  # (B, T, d_v)
        beta = torch.sigmoid(self.proj_beta(x))  # (B, T, 1)

        y_out = torch.zeros(b, t, self.d_v, device=device)
        surprises_logged = 0

        for batch_idx in range(b):
            s_state = torch.zeros(self.d_k, self.d_v, device=device)
            reservoir = ErrorReservoir(capacity=self.reservoir_capacity, d_k=self.d_k, d_v=self.d_v, device=device)

            for step in range(t):
                q_t = q[batch_idx, step]  # (d_k,)
                k_t = k[batch_idx, step]  # (d_k,)
                v_t = v[batch_idx, step]  # (d_v,)
                b_t = beta[batch_idx, step]  # (1,)

                # 1. Delta error computation: e_t = v_t - S_{t-1} k_t
                v_pred = torch.matmul(k_t, s_state)  # (d_v,)
                err = v_t - v_pred
                err_norm = torch.norm(err, p=2).item()

                # 2. Continuous associative state update
                s_state = s_state + b_t * torch.outer(k_t, err)

                # 3. Surprise-gated reservoir write
                if err_norm > self.surprise_threshold:
                    reservoir.push(k_t, err)
                    surprises_logged += 1

                # 4. Decoupled dual readout
                y_fast = torch.matmul(q_t, s_state)  # (d_v,)
                y_slow = reservoir.query(q_t)        # (d_v,)
                y_out[batch_idx, step] = y_fast + y_slow

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
