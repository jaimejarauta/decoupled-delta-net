#!/usr/bin/env python3
"""Train on MQAR and compare matched arms: Decoupled Delta-Net vs plain DeltaNet (reservoir disabled).
Question: does the surprise reservoir raise held-out recall at equal size, data and steps?
"""

import argparse
import json
import os
import subprocess
import time

import torch
import torch.nn.functional as F

from decoupled_delta.model import DecoupledDeltaModel
from decoupled_delta.tasks import generate_mqar_batch
from decoupled_delta.metrics import compute_mqar_accuracy


def git_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def train_arm(name: str, threshold: float, args, device, relative: bool = False) -> dict:
    torch.manual_seed(args.seed)
    model = DecoupledDeltaModel(vocab_size=args.vocab_size, d_model=args.d_model, d_k=args.d_k, d_v=args.d_k,
                                num_layers=args.num_layers, surprise_threshold=threshold,
                                relative=relative).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)
    t0 = time.time()
    for step in range(args.steps):
        b = generate_mqar_batch(args.seq_len, args.num_pairs, args.vocab_size, args.batch_size, seed=args.seed * 100_000 + step)
        logits, telem = model(b["input_ids"].to(device))
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size), b["labels"].to(device).reshape(-1), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 100 == 0:
            print(f"  [{name}] step {step + 1}/{args.steps} loss {loss.item():.3f} "
                  f"surprise {telem['layer_0_surprise_rate'] * 100:.1f}%")

    model.eval()
    with torch.no_grad():  # held-out seed range never used in training
        b = generate_mqar_batch(args.seq_len, args.num_pairs, args.vocab_size, args.eval_batch, seed=10**9 + args.seed)
        logits, telem = model(b["input_ids"].to(device))
        acc = compute_mqar_accuracy(logits, b["labels"].to(device))
    return {"arm": name, "surprise_threshold": threshold, "relative": relative, "eval_accuracy": acc,
            "eval_surprise_rate": telem["layer_0_surprise_rate"], "final_train_loss": loss.item(),
            "train_sec": time.time() - t0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-k", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threshold", type=float, default=1.5,
                   help="Decoupled arm: surprise if err^2 > threshold x running mean err^2")
    p.add_argument("--absolute", action="store_true", help="Use an absolute RMS threshold instead (failed 2026-10-26)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--output", type=str, default="results/mqar_train_manifest.json")
    args = p.parse_args()
    device = torch.device(args.device)

    # Gate (2026-10-26): reservoir arm must beat plain DeltaNet by >= 5 points of held-out recall
    arms = [train_arm("deltanet", float("inf"), args, device),
            train_arm("decoupled", args.threshold, args, device, relative=not args.absolute)]
    margin = arms[1]["eval_accuracy"] - arms[0]["eval_accuracy"]
    gate = margin >= 0.05
    for a in arms:
        print(f"{a['arm']:<10} recall {a['eval_accuracy'] * 100:.1f}% | surprise {a['eval_surprise_rate'] * 100:.1f}% "
              f"| {a['train_sec']:.0f}s")
    print(f"Margin {margin * 100:+.1f} pts (need >= +5.0) -> {'PASS' if gate else 'FAIL'}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"git_commit": git_hash(), "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "config": vars(args), "arms": arms, "margin": margin, "gate_passed": gate}, f, indent=2)
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
