#!/usr/bin/env python3
"""Single-Command Self-Healing Google Colab Runner for Decoupled Delta-Net.
Executes the MQAR Capacity Scaling Benchmark across K=8..64 pairs and T=2k..8k context.
Followed by automatic W&B cloud upload and VM teardown.
"""

import os
import sys
import signal
import subprocess

def main():
    print("=" * 70)
    print("DECOUPLED DELTA-NET: MQAR CAPACITY SCALING BENCHMARK")
    print("=" * 70)

    # 1. Credit Guard: Strict 60-second timeout for any interactive prompt
    def timeout_handler(signum, frame):
        print("\n[Credit Guard Alert] Interactive auth timed out (60s without input)!")
        print("[Credit Guard Alert] Unassigning Colab GPU immediately to prevent idle burn...")
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception:
            pass
        sys.exit(1)

    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)

    # 2. Multi-Candidate Colab Secrets Search
    wandb_key = os.environ.get("WANDB_API_KEY")
    if not wandb_key:
        try:
            from google.colab import userdata
            candidate_keys = [
                'WANDB_API_KEY', 'wandb_api_key', 'WANDB_KEY', 'wandb_key',
                'wandb', 'WANDB', 'wandb_token', 'WANDB_TOKEN', 'API_KEY', 'api_key'
            ]
            for k in candidate_keys:
                try:
                    val = userdata.get(k)
                    if val and len(val.strip()) >= 20:
                        wandb_key = val.strip()
                        print(f"[W&B Guard] Detected API key from Colab Secrets: '{k}'")
                        break
                except Exception:
                    continue
        except Exception:
            pass

    # Check ~/.netrc as fallback
    netrc_path = os.path.expanduser('~/.netrc')
    if not wandb_key and os.path.exists(netrc_path):
        try:
            with open(netrc_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2 and parts[0] == 'password' and len(parts[1]) >= 20:
                        wandb_key = parts[1]
                        print("[W&B Guard] Restored API key from ~/.netrc")
                        break
        except Exception:
            pass

    # 3. Authenticate or Fallback to Offline Mode
    if wandb_key:
        os.environ['WANDB_API_KEY'] = wandb_key
        os.environ['WANDB_MODE'] = 'online'
        try:
            import wandb
            wandb.login(key=wandb_key, relogin=False)
            print("[W&B Guard] Authenticated successfully in online mode.")
        except Exception as e:
            print(f"[W&B Guard Warning] Login notice: {e}. Will sync after completion.")
    else:
        print("[W&B Guard Notice] No W&B key detected. Running in offline mode (never hangs).")
        os.environ['WANDB_MODE'] = 'offline'

    # Clear interactive timeout alarm now that auth is resolved
    if hasattr(signal, 'SIGALRM'):
        signal.alarm(0)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    repo_src = os.path.join(repo_root, "src")

    # 4. Install Dependencies & Local Package
    print("\n[Runner] Ensuring required Python dependencies and package installation...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-e", repo_root,
        "wandb", "pyyaml"
    ], check=True)

    bench_script = os.path.join(repo_root, "scripts", "bench_mqar_scale.py")
    wandb_mode = os.environ.get("WANDB_MODE", "online")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_src}:{env.get('PYTHONPATH', '')}"

    # 5. Run Stage 2 MQAR Benchmark
    cmd = [
        sys.executable, bench_script,
        "--lengths", "2048", "4096", "8192",
        "--pairs", "8", "16", "32", "64",
        "--wandb",
        "--wandb-project", "research-projekts",
        "--wandb-mode", wandb_mode,
        "--wandb-name", "decoupled-delta-mqar-scale-l4"
    ]
    print(f"\n[Runner] Executing: {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, env=env)

    # 6. Flush & Sync Offline W&B Runs
    print("\n[Runner] Syncing all W&B runs to cloud...")
    subprocess.run("wandb sync --include-offline $(find /tmp/wandb ./wandb -type d -name 'offline-run-*' 2>/dev/null) 2>/dev/null || true", shell=True)

    # 7. Credit Guard: Teardown VM
    try:
        from google.colab import runtime
        print("\n[Credit Guard] Run complete. Unassigning GPU to save Colab credits...")
        runtime.unassign()
    except Exception as e:
        print(f"[Credit Guard] Teardown notice: {e}")

    sys.exit(ret.returncode)

if __name__ == "__main__":
    main()
