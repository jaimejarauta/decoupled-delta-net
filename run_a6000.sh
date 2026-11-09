#!/usr/bin/env bash
# correction-memory on a Linux + NVIDIA box (A6000 / Blackwell): venv, installs, then the full resumable campaign in the
# background (pytest pre-flight -> pilot -> memories -> dev -> test -> gate). Re-running resumes from $CM_RESULTS.
# Usage:  bash run_a6000.sh      (HF_TOKEN optional: all checkpoints and the lerobot/libero demos are public)
set -euo pipefail
cd "$(dirname "$0")"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CM_RESULTS="${CM_RESULTS:-$PWD/results}" CM_MAX_HOURS="${CM_MAX_HOURS:-24}"   # README amendment 2026-11-09
mkdir -p "$CM_RESULTS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null; then uv venv -q -p 3.12 .venv; else "${PYTHON:-python3.12}" -m venv .venv; fi
fi
. .venv/bin/activate
PIP="python -m pip"; command -v uv >/dev/null && PIP="uv pip"
$PIP install -q pip setuptools wheel cmake

# egl_probe / hf-egl-probe (pulled by hf-libero, robomimic) often fail to build; they only pick the EGL GPU -> stub them
for spec in "hf-egl-probe:egl_probe" "egl_probe:"; do
  dist=${spec%%:*}; mod=${spec#*:}; d=.egl_stubs/$dist; mkdir -p "$d"
  mods='[]'; [ -n "$mod" ] && { mods="[\"$mod\"]"; printf 'def get_available_devices():\n    return [0]\n' > "$d/$mod.py"; }
  printf '[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"\n[project]\nname = "%s"\nversion = "1.0.2"\n[tool.setuptools]\npy-modules = %s\n' "$dist" "$mods" > "$d/pyproject.toml"
  $PIP install -q --no-build-isolation "./$d"
done
$PIP install -q -r requirements.txt
python - <<'PY'   # silence robosuite's per-env "no private macro file" warning
import importlib.util, os, shutil
d = os.path.dirname(importlib.util.find_spec("robosuite").origin)
os.path.exists(f"{d}/macros_private.py") or shutil.copy(f"{d}/macros.py", f"{d}/macros_private.py")
PY
python -c "import torch, lerobot; assert torch.cuda.is_available(); print('torch', torch.__version__, '| lerobot', lerobot.__version__)"

nohup python -u run_all.py >> "$CM_RESULTS/run.log" 2>&1 &
echo "started pid $! -> follow: tail -f $CM_RESULTS/run.log   (result: $CM_RESULTS/RESULTS.md)"
