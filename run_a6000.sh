#!/usr/bin/env bash
# vla-reachability on a Linux + NVIDIA box (A6000 / Blackwell): venv, installs, then the full resumable campaign in the
# background (pytest pre-flight -> pilot -> main -> RESULTS.md). Re-running resumes from $VR_RESULTS.
# Usage:  HF_TOKEN=hf_xxx bash run_a6000.sh          (token needs access to google/paligemma-3b-pt-224, for pi05)
set -euo pipefail
cd "$(dirname "$0")"
: "${HF_TOKEN:?export HF_TOKEN=hf_... first (Hugging Face read token with access to google/paligemma-3b-pt-224)}"
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export VR_RESULTS="${VR_RESULTS:-$PWD/results}" VR_BUDGET_CU="${VR_BUDGET_CU:-150}"   # README amendment 2026-11-09
mkdir -p "$VR_RESULTS"
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

nohup python -u run_all.py >> "$VR_RESULTS/run.log" 2>&1 &
echo "started pid $! -> follow: tail -f $VR_RESULTS/run.log   (result: $VR_RESULTS/RESULTS.md)"
