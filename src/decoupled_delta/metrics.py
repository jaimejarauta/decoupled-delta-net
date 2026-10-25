"""Metrics and evaluation benchmarks for Decoupled Delta-Net."""

import time
from typing import Dict
import torch


def compute_mqar_accuracy(logits: torch.Tensor, target_val: torch.Tensor) -> float:
    """Computes associative recall top-1 accuracy at query position."""
    # logits: (B, T, vocab_size), target_val: (B,)
    preds = torch.argmax(logits[:, -1, :], dim=-1)
    correct = (preds == target_val).float().sum().item()
    return float(correct / len(target_val))


def benchmark_associative_throughput(
    model: torch.nn.Module,
    seq_len: int = 2048,
    batch_size: int = 2,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """Measures streaming inference throughput and surprise statistics."""
    model.eval()
    x = torch.randint(10, 200, (batch_size, seq_len), device=device)

    # Warmup
    with torch.no_grad():
        _ = model(x[:, :64])

    t0 = time.perf_counter()
    with torch.no_grad():
        logits, telem = model(x)
    elapsed = time.perf_counter() - t0

    tokens = seq_len * batch_size
    tokens_per_sec = tokens / elapsed
    ms_per_token = (elapsed / tokens) * 1000.0

    return {
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token,
        "elapsed_sec": elapsed,
        "seq_len": seq_len,
        "surprise_rate": telem.get("layer_0_surprise_rate", 0.0),
    }
