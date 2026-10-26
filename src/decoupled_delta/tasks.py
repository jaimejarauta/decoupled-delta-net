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
    """Multi-Query Associative Recall (Zoology layout).
    K unique key-value pairs fill the first 2K tokens; every key is queried once at a random later slot,
    followed by its value. Keys come from [1, V/2), values from [V/2, V), filler is 0.
    labels[b, p] is the value to predict at input position p (a query key), -100 elsewhere.
    """
    rng = random.Random(seed)
    half = vocab_size // 2
    assert num_pairs < half and 4 * num_pairs <= seq_len, "too many pairs for vocab or length"
    input_ids_list, labels_list, target_vals = [], [], []

    for _ in range(batch_size):
        seq = [0] * seq_len
        lab = [-100] * seq_len
        keys = rng.sample(range(1, half), num_pairs)
        vals = [rng.randrange(half, vocab_size) for _ in keys]
        for i, (k, v) in enumerate(zip(keys, vals)):
            seq[2 * i], seq[2 * i + 1] = k, v

        # Query slots: even offsets after the context, so key/value pairs never overlap
        slots = sorted(rng.sample(range(2 * num_pairs, seq_len - 1, 2), num_pairs))
        order = list(range(num_pairs))
        rng.shuffle(order)
        for pos, i in zip(slots, order):
            seq[pos], seq[pos + 1] = keys[i], vals[i]
            lab[pos + 1] = vals[i]

        input_ids_list.append(seq[:-1])
        labels_list.append(lab[1:])
        target_vals.append(vals[order[-1]])

    return {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long),
        "target_val": torch.tensor(target_vals, dtype=torch.long),  # value of the last query
        "seq_len": seq_len,
    }
