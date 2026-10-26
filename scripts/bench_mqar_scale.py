#!/usr/bin/env python3
"""Multi-Scale Multi-Query Associative Recall (MQAR) Benchmark for Decoupled Delta-Net.
Evaluates associative capacity across increasing pair counts (K=8, 16, 32, 64)
and sequence lengths (T=2048, 4096, 8192).
"""

import argparse
import os
import sys
import time
import json
import torch
import torch.nn.functional as F

repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if repo_src not in sys.path:
    sys.path.insert(0, repo_src)

from decoupled_delta.model import DecoupledDeltaModel
from decoupled_delta.tasks import generate_mqar_batch

def main():
    parser = argparse.ArgumentParser(description="MQAR Multi-Scale Benchmark")
    parser.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096, 8192])
    parser.add_argument("--pairs", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-k", type=int, default=32)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="research-projekts")
    parser.add_argument("--wandb-name", type=str, default="decoupled-delta-mqar-scale")
    parser.add_argument("--wandb-mode", type=str, default="online")
    args = parser.parse_args()

    use_wandb = args.wandb or os.environ.get("WANDB_MODE") in ["online", "offline"]
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                mode=args.wandb_mode,
                config=vars(args)
            )
        except Exception as e:
            print(f"[W&B Warning] Init skipped: {e}")
            use_wandb = False

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("=" * 70)
    print("DECOUPLED DELTA-NET: MQAR CAPACITY SCALING BENCHMARK")
    print(f"Device: {device} | Architecture: {args.num_layers} layers x d_model={args.d_model}")
    print("=" * 70)

    model = DecoupledDeltaModel(
        vocab_size=256,
        d_model=args.d_model,
        d_k=args.d_k,
        d_v=args.d_v,
        num_layers=args.num_layers
    ).to(device)
    model.eval()

    results = []
    print(f"{'Context (T)':<12} | {'Pairs (K)':<10} | {'Throughput (tok/s)':<18} | {'Latency (ms/tok)':<16} | {'Surprise Rate':<14} | {'Status'}")
    print("-" * 85)

    with torch.no_grad():
        for seq_len in args.lengths:
            for num_pairs in args.pairs:
                batch = generate_mqar_batch(seq_len=seq_len, num_pairs=num_pairs, vocab_size=256, batch_size=2, seed=42)
                input_ids = batch["input_ids"].to(device)

                # Warmup
                _ = model(input_ids[:, :64])

                t0 = time.perf_counter()
                logits, telemetry = model(input_ids)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - t0

                total_tokens = input_ids.numel()
                throughput = total_tokens / max(1e-6, elapsed)
                latency_ms = (elapsed / total_tokens) * 1000.0

                surprise_rate = telemetry.get("surprise_rate", 0.0) * 100.0
                status = "PASSED" if throughput >= 150.0 and latency_ms <= 10.0 else "FAILED"

                print(f"{seq_len:<12,d} | {num_pairs:<10d} | {throughput:<18.1f} | {latency_ms:<16.3f} | {surprise_rate:<13.1f}% | {status}")

                row = {
                    "seq_len": seq_len,
                    "num_pairs": num_pairs,
                    "throughput_tok_s": throughput,
                    "latency_ms": latency_ms,
                    "surprise_rate": surprise_rate,
                    "status": status
                }
                results.append(row)

                if use_wandb:
                    wandb.log(row)

    print("=" * 85)
    os.makedirs("results", exist_ok=True)
    with open("results/mqar_scale_manifest.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to results/mqar_scale_manifest.json")

if __name__ == "__main__":
    main()
