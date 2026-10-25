#!/usr/bin/env bash
# Phase-2 bootstrap (Wan / FLUX / MoGe sessions). CLAUDE.md §1: Phase 1 and Phase 2 are memory-disjoint and, measured on
# 2026-10-07, ENVIRONMENT-disjoint too: scripts/bootstrap.sh installs the tracker's requirements, which pin torch 2.6 under a
# Colab image whose torchvision/torchaudio/torchao are built for torch 2.11, and every diffusers/transformers import then
# fails (`aoti_torch_abi_version`, `FqnToConfig`, `ResolvedRevision`). So a Phase-2 session runs THIS script and never
# scripts/bootstrap.sh; a Phase-1 session the reverse. Same contract as bootstrap.sh for secrets: env only, never printed.
#
#   Fresh runtime -> notebook cell 0 (mount Drive, clone, cd into the repo) -> bash scripts/bootstrap_phase2.sh
set -euo pipefail
DRIVE="${DRIVE:-/content/drive/MyDrive}"   # optional: secrets are read from here when it is mounted
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

load_secret () {
  local file="$1" var="$2" line val
  [ -f "$file" ] || { echo "FATAL: missing secret file $file"; exit 1; }
  line="$(grep -m1 -E '^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=' "$file")" || { echo "FATAL: no KEY=VALUE line in $file"; exit 1; }
  val="${line#*=}"; val="${val#"${val%%[![:space:]]*}"}"; val="${val%"${val##*[![:space:]]}"}"
  val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
  export "$var"="$val"
}
[ -n "${GH_TOKEN:-}" ] || [ ! -f "$DRIVE/secrets/gh_token.env" ] || load_secret "$DRIVE/secrets/gh_token.env" GH_TOKEN
[ -n "${HF_TOKEN:-}" ] || [ ! -f "$DRIVE/secrets/hf_token.env" ] || load_secret "$DRIVE/secrets/hf_token.env" HF_TOKEN
export GH_TOKEN="${GH_TOKEN:-}" HF_TOKEN="${HF_TOKEN:-}"   # a public clone needs no GitHub token; Module 2 needs an HF token for FLUX.1-dev
export GH_TOKEN HF_TOKEN

if [ -n "$GH_TOKEN" ]; then
  git config --global credential.helper 'store --file /tmp/.git-credentials'
  printf 'https://x-access-token:%s@github.com\n' "$GH_TOKEN" > /tmp/.git-credentials
  chmod 600 /tmp/.git-credentials
fi
# Pull, then re-exec the PULLED copy of this script: bash reads a script as it runs, so a pull that changes this file mid-run
# would otherwise leave the old copy executing (measured 2026-10-07: the video-download step was pulled in and not run).
if [ "${1:-}" != "--pulled" ]; then
  before="$(sha256sum "$0" | cut -c1-16)"
  git pull --ff-only 2>/dev/null || true
  after="$(sha256sum "$0" | cut -c1-16)"
  if [ "$before" != "$after" ]; then
    echo "bootstrap_phase2.sh changed by the pull; restarting with the new copy"
    exec bash "$0" --pulled
  fi
fi

# ---- Phase-2 libraries on Colab's NATIVE torch (never pin torch here) ----
pip uninstall -y -q torchaudio 2>/dev/null || true          # its binary does not match the image's torch; transformers imports it if present
pip install -q -U "diffusers>=0.40" "transformers>=4.57" "accelerate>=1.0" "huggingface_hub>=1.0" ftfy pyyaml scipy \
    opencv-python-headless imageio imageio-ffmpeg
# torchao: the image's copy is older than what diffusers >= 0.40 imports (`FqnToConfig`, measured 2026-10-07). Upgrade; if no wheel
# matches the image's torch, remove it (diffusers then skips it; the S-mem probe records "FP8 unavailable" and runs bf16 only).
pip install -q -U "torchao>=0.12" 2>/dev/null || pip uninstall -y -q torchao || true
# MoGe-2 (S3 depth from x_hat_0, the refresh): the same adapter as S-int step 1; not on the image. Preserves torch (no deps pulled).
python -c "import utils3d" 2>/dev/null || pip install -q --no-deps "git+https://github.com/EasternJournalist/utils3d.git"   # MoGe's own geometry helper, not on PyPI
python -c "import moge" 2>/dev/null || pip install -q --no-deps "git+https://github.com/microsoft/MoGe.git"
python -c "import moge.model.v2" || echo "WARN: moge import failed; install the named package alone (plain pip, it will not touch torch)"
# ---- reference videos -> local disk, as bootstrap.sh does (S2 reads the source clips at Wan's frame size) ----
eval "$(python - <<'PY'
import yaml
c = yaml.safe_load(open("configs/stage_a.yaml", encoding="utf-8"))
print(f"export HF_DATASET_ID='{c['repo']['hf_dataset_id']}'")
print(f"export DATA_DIR='{c['paths']['data_dir']}'")
PY
)"
mkdir -p "$DATA_DIR"
if [ "${SKIP_VIDEOS:-0}" != "1" ]; then
python - <<'PY' || echo "WARN: the reference-video dataset was not downloaded (private or unreachable); put the clips under $DATA_DIR yourself"
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=os.environ["HF_DATASET_ID"], repo_type="dataset", local_dir=os.environ["DATA_DIR"])
print("videos ->", p)
PY
fi
python - <<'PY'
import torch, diffusers, transformers
from diffusers import WanImageToVideoPipeline, FluxControlPipeline  # noqa: F401
print(f"phase-2 env: torch {torch.__version__} | diffusers {diffusers.__version__} | transformers {transformers.__version__} | "
      f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'} | imports ok")
PY
