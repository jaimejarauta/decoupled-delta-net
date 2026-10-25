"""Synthetic associative recall benchmarks for Decoupled Delta-Net."""

import random
from typing import Dict
import torch


def generate_mqar_batch(
    seq_len: int = 1024,
    num_pairs: int = 8,
    vocab_size: int = 256,
    batch_size: int = 2,
    seed: int = 2026,
) -> Dict[str, torch.Tensor]:
    """Generates Multi-Query Associative Recall sequence batch."""
    rng = random.Random(seed)
    input_ids_list = []
    target_vals = []

    for b in range(batch_size):
        seq = [rng.randint(80, vocab_size - 1) for _ in range(seq_len)]
        pairs = {}
        for p in range(num_pairs):
            k = rng.randint(1, 35)
            v = rng.randint(36, 75)
            pairs[k] = v
            pos = rng.randint(20, int(seq_len * 0.70))
            seq[pos] = k
            seq[pos + 1] = v

        # Place single query token at the end
        query_k = rng.choice(list(pairs.keys()))
        target_v = pairs[query_k]
        seq[-2] = query_k
        seq[-1] = target_v

        input_ids_list.append(seq[:-1])
        target_vals.append(target_v)

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "target_val": torch.tensor(target_vals, dtype=torch.long),
        "seq_len": seq_len,
    }
