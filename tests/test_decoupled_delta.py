"""Unit tests for Decoupled Delta-Net."""

import torch
from decoupled_delta.model import DecoupledDeltaCell, DecoupledDeltaBlock, DecoupledDeltaModel
from decoupled_delta.tasks import generate_mqar_batch


def test_decoupled_delta_cell():
    cell = DecoupledDeltaCell(d_model=32, d_k=16, d_v=16)
    x = torch.randn(2, 16, 32)
    y, telem = cell(x)
    assert y.shape == (2, 16, 32)
    assert "surprise_rate" in telem


def test_decoupled_delta_model():
    model = DecoupledDeltaModel(vocab_size=100, d_model=32, d_k=16, d_v=16, num_layers=1)
    input_ids = torch.randint(0, 100, (2, 32))
    logits, telem = model(input_ids)
    assert logits.shape == (2, 32, 100)


def test_decoupled_delta_backward():
    """Verify autograd backward pass works cleanly without in-place modification errors."""
    model = DecoupledDeltaModel(vocab_size=100, d_model=32, d_k=16, d_v=16, num_layers=2)
    input_ids = torch.randint(0, 100, (2, 16))
    logits, _ = model(input_ids)
    loss = logits.sum()
    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_mqar_task_generation():
    batch = generate_mqar_batch(seq_len=64, num_pairs=4, vocab_size=100, batch_size=2)
    assert batch["input_ids"].shape == (2, 63)
    assert batch["target_val"].shape == (2,)



def test_mqar_unique_keys_and_multi_query_labels():
    b = generate_mqar_batch(seq_len=128, num_pairs=8, vocab_size=128, batch_size=16, seed=3)
    ids, labels = b["input_ids"], b["labels"]
    assert ids.shape == labels.shape == (16, 127)
    for row_ids, row_lab in zip(ids, labels):
        kv = {}
        for t in range(0, 16, 2):  # pairs occupy the first 2K tokens
            k, v = int(row_ids[t]), int(row_ids[t + 1])
            assert k not in kv
            kv[k] = v
        q = (row_lab != -100).nonzero().squeeze(-1)
        assert len(q) == 8  # every key is queried once
        for p in q:
            # labels[p] is the next token's target: the value of the key at input position p
            assert int(row_lab[p]) == kv[int(row_ids[p])]
    assert b["target_val"].shape == (16,)


def test_mqar_accuracy_uses_all_queries():
    from decoupled_delta.metrics import compute_mqar_accuracy
    labels = torch.full((1, 5), -100)
    labels[0, 1], labels[0, 3] = 7, 9
    logits = torch.zeros(1, 5, 10)
    logits[0, 1, 7] = 1.0
    logits[0, 3, 2] = 1.0
    assert compute_mqar_accuracy(logits, labels) == 0.5


def test_relative_surprise_is_scale_invariant():
    torch.manual_seed(0)
    cell = DecoupledDeltaCell(d_model=32, d_k=16, d_v=16, surprise_threshold=1.2, relative=True)
    x = torch.randn(2, 64, 32)
    with torch.no_grad():
        _, t1 = cell(x)
        cell.proj_v.weight.mul_(100.0)  # errors 100x larger
        _, t2 = cell(x)
    assert 0.0 < t1["surprise_rate"] < 0.9
    assert abs(t1["surprise_rate"] - t2["surprise_rate"]) < 1e-6
