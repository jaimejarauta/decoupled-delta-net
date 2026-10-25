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


def test_mqar_task_generation():
    batch = generate_mqar_batch(seq_len=64, num_pairs=4, vocab_size=100, batch_size=2)
    assert batch["input_ids"].shape == (2, 63)
    assert batch["target_val"].shape == (2,)

