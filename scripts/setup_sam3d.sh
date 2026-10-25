#!/usr/bin/env bash
# One-time Colab setup for the SAM 3D reconstructor probe (skills/CC-TASK-reconstructor-probe.md).
# Idempotent; records every outcome; never hides a failure. Run AFTER bootstrap (HF_TOKEN in env).
#
#   bash scripts/setup_sam3d.sh            # both variants
#   bash scripts/setup_sam3d.sh body|objects
#
# MEASURED 2026-09-11 (A100-40GB, Colab python 3.13.15, torch 2.6.0+cu124):
#   * both gated downloads succeed once HF access is granted (token from bootstrap);
#   * sam-3d-body is not a pip package (no setup.py) — it is used from its repo dir;
#   * sam-3d-objects pins torch 2.5.1+cu121, xformers 0.0.28.post3, spconv-cu121, kaolin
#     0.17.0, flash_attn 2.8.3, pytorch3d@git, auto_gptq 0.7.1 — auto_gptq has no build
#     for py3.13, so pip refuses the whole set. The vendor's supported route is a conda env
#     (environments/default.yml: python 3.11 + cudatoolkit 12.1 + compilers). We build
#     exactly that with micromamba and run the probe with THAT env's python:
#         $OBJ_PY scripts/probe_sam3d.py --p0 --variant objects
#   * the env is large and slow to build (pytorch3d compiles); it is tarred to Drive after
#     the first success and restored on later sessions (CLAUDE.md §5: >2 min -> checkpoint).
set -uo pipefail
WHICH="${1:-both}"
OBJ_DIR="${SAM3D_OBJECTS_DIR:-/content/sam-3d-objects}"
BODY_DIR="${SAM3D_BODY_DIR:-/content/sam-3d-body}"
MM_ROOT="${MAMBA_ROOT_PREFIX:-/content/micromamba}"
ENV_NAME=sam3d-objects
OBJ_PY="$MM_ROOT/envs/$ENV_NAME/bin/python"
DRIVE_ROOT="${DRIVE_ROOT:-/content/drive/MyDrive/motion_transfer}"
ENV_TAR="$DRIVE_ROOT/envs/${ENV_NAME}.tar"
# log lives on DRIVE so it survives a dead session (2026-09-11: a 155-min run's log was lost
# with the VM); /content only if Drive is not mounted
if [ -d "$DRIVE_ROOT" ]; then mkdir -p "$DRIVE_ROOT/envs"; LOG="$DRIVE_ROOT/envs/setup_sam3d_$(date -u +%Y%m%dT%H%M%SZ).log"; else LOG=/content/sam3d_setup.log; fi
: > "$LOG"; ln -sf "$LOG" /content/sam3d_setup.log 2>/dev/null || true
say () { echo "[setup_sam3d] $*" | tee -a "$LOG"; }
[ -n "${HF_TOKEN:-}" ] || say "WARN: HF_TOKEN not in env (run the bootstrap cell first); gated downloads will fail"
python -c "import sys, torch; print('system python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.version.cuda)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$LOG" || say "no GPU"

# ---------------------------------------------------------------- SAM 3D Body (system python)
if [ "$WHICH" = both ] || [ "$WHICH" = body ]; then
  say "== SAM 3D Body =="
  [ -d "$BODY_DIR/.git" ] || git clone -q https://github.com/facebookresearch/sam-3d-body.git "$BODY_DIR"
  # INSTALL.md deps, one at a time so a failing build is NAMED, not buried
  for pkg in pytorch-lightning pyrender yacs scikit-image einops timm dill rich hydra-core \
             hydra-submitit-launcher hydra-colorlog pyrootutils webdataset chump networkx roma \
             joblib appdirs cython jsonlines xtcocotools loguru optree fvcore pycocotools; do
    pip install -q "$pkg" >/dev/null 2>&1 || say "body dep FAILED: $pkg (recorded)"
  done
  # detectron2 is imported by sam_3d_body for its config/visualisation modules only; its CUDA ops are never called on the Body path.
  # Built CPU-only on purpose (2026-10-19): Colab's system nvcc moved to CUDA 13.0 while torch is 12.4-built, and torch's extension
  # builder refuses the mismatch; with no CUDA visible and no CUDA_HOME the build takes the C++-only branch and needs no nvcc.
  pip install -q ninja >/dev/null 2>&1 || true
  if ! CUDA_VISIBLE_DEVICES="" CUDA_HOME=/nonexistent FORCE_CUDA=0 pip install -q 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps >/tmp/detectron2_pip.log 2>&1; then
    say "body: detectron2 install FAILED (recorded); last lines of pip:"; tail -n 25 /tmp/detectron2_pip.log | tee -a "$LOG"
  fi
  python -c "import detectron2" 2>/dev/null && say "body: detectron2 importable" || say "body: detectron2 NOT importable in the system python (the Body R2 cannot run)"
  pip install -q git+https://github.com/microsoft/MoGe.git >/dev/null 2>&1 || say "body: MoGe install FAILED (recorded)"
  pip install -q 'git+https://github.com/facebookresearch/sam2.git' >/dev/null 2>&1 || say "body: sam2 install FAILED (recorded)"
  # Generation + Body stack, MEASURED working together 2026-09-14 (P4 gen + Body ran on it):
  # torch 2.6.0+cu124 (Colab base, not reinstalled). The earlier 'huggingface-hub<1.0, transformers<5'
  # pin broke on a fresh session (diffusers 0.40 needs hub 1.x; hub 1.x drops is_offline_mode that
  # transformers 4.x imports). torchaudio's ABI mismatched this torch (undefined aoti_torch_abi_version)
  # and is imported by transformers when present; nothing here uses it -> removed.
  pip install -q "transformers==5.17.0" "diffusers==0.40.0" "huggingface_hub==1.31.0" "accelerate==1.15.0"                  "safetensors==0.8.0" "numpy==2.2.5" sentencepiece protobuf >/dev/null 2>&1 || say "generation stack pin FAILED (recorded)"
  pip uninstall -q -y torchaudio >/dev/null 2>&1 || true
  python -c "import torch; v=torch.__version__; print('[setup_sam3d] torch', v, '' if v.startswith('2.6.0') else '(WARNING: stack was pinned against 2.6.0+cu124)')" 2>&1 | tee -a "$LOG"
  python - <<'PY' 2>&1 | tee -a "$LOG"
from huggingface_hub import snapshot_download
try:
    print("[setup_sam3d] body weights ->", snapshot_download(repo_id="facebook/sam-3d-body-dinov3", repo_type="model"))
except Exception as e:
    print(f"[setup_sam3d] body weights NOT downloaded: {type(e).__name__}: {str(e)[:160]}")
PY
  (cd "$BODY_DIR" && python -c "import sam_3d_body; print('[setup_sam3d] sam_3d_body imports from repo dir')" 2>&1 | tail -1 | tee -a "$LOG")
fi

# ---------------------------------------------------------------- SAM 3D Objects (own env)
if [ "$WHICH" = both ] || [ "$WHICH" = objects ]; then
  say "== SAM 3D Objects (micromamba env, vendor recipe: doc/setup.md) =="
  [ -d "$OBJ_DIR/.git" ] || git clone -q https://github.com/facebookresearch/sam-3d-objects.git "$OBJ_DIR"
  mkdir -p "$MM_ROOT/bin"
  if [ ! -x "$MM_ROOT/bin/micromamba" ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C "$MM_ROOT" bin/micromamba
  fi
  export MAMBA_ROOT_PREFIX="$MM_ROOT"
  MM="$MM_ROOT/bin/micromamba"
  if [ ! -x "$OBJ_PY" ] && [ -f "$ENV_TAR" ]; then
    # a tar interrupted mid-write (session death; 2026-09-12 the log ended at "packing")
    # restores a broken env silently. Trust it only with its completion marker; a tar
    # without one is verified end-to-end once (tar -tf fails on truncation) and then marked.
    if [ ! -f "$ENV_TAR.ok" ]; then
      say "verifying $ENV_TAR (no completion marker) — reads the whole archive once"
      if tar -tf "$ENV_TAR" >/dev/null 2>&1; then touch "$ENV_TAR.ok"; say "archive intact; marked"; else
        say "archive TRUNCATED — deleting; the env will be rebuilt (~25 min with the cached wheel)"; rm -f "$ENV_TAR"; fi
    fi
    if [ -f "$ENV_TAR.ok" ]; then
      say "restoring env from Drive: $ENV_TAR"
      mkdir -p "$MM_ROOT/envs" && tar -xf "$ENV_TAR" -C "$MM_ROOT/envs"
    fi
  fi
  if [ ! -x "$OBJ_PY" ]; then
    say "creating env $ENV_NAME from environments/default.yml (python 3.11 + cudatoolkit 12.1) — minutes"
    "$MM" create -y -q -n "$ENV_NAME" -f "$OBJ_DIR/environments/default.yml" 2>&1 | tail -2 | tee -a "$LOG"
  fi
  [ -x "$OBJ_PY" ] || { say "env python missing at $OBJ_PY — env creation failed; see $LOG"; exit 1; }
  "$OBJ_PY" -c "import sys; print('[setup_sam3d] env python', sys.version.split()[0])" | tee -a "$LOG"
  export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
  export MAX_JOBS="${MAX_JOBS:-$(nproc)}"       # Colab T4/A100 VMs have 2-12 vCPUs; more jobs than cores only thrashes
  WHEEL_DIR="$DRIVE_ROOT/envs/wheels"; mkdir -p "$WHEEL_DIR"
  step () {  # step <label> <cmd...>: run inside the env, record pass/fail, keep going
    local label="$1"; shift
    if (cd "$OBJ_DIR" && "$@" >>"$LOG" 2>&1); then say "objects: $label OK"; else say "objects: $label FAILED (see $LOG)"; fi
  }
  step "[dev] (torch 2.5.1+cu121, xformers, spconv, auto_gptq ...)" "$OBJ_PY" -m pip install -q -e '.[dev]'
  # flash_attn 2.8.3: PyPI ships SOURCE ONLY (measured 2026-09-11: a from-source build ran
  # > 55 min on the Colab CPU and was still going). The project's GitHub release carries
  # prebuilt wheels; pick the one matching the env's torch minor + C++ ABI so [p3d] below
  # finds it satisfied and only has to compile pytorch3d.
  FA_VER=2.8.3
  FA_TORCH=$("$OBJ_PY" -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(v[0]+'.'+v[1])")
  FA_ABI=$("$OBJ_PY" -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
  FA_PY=$("$OBJ_PY" -c "import sys; print(f'cp{sys.version_info[0]}{sys.version_info[1]}')")
  FA_WHL="flash_attn-${FA_VER}+cu12torch${FA_TORCH}cxx11abi${FA_ABI}-${FA_PY}-${FA_PY}-linux_x86_64.whl"
  say "flash_attn prebuilt wheel: $FA_WHL"
  step "flash_attn ${FA_VER} prebuilt wheel"                          "$OBJ_PY" -m pip install -q        "https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VER}/${FA_WHL}"
  # ---- pytorch3d (the only compile; measured 2026-09-11: ~2 h on a 2-vCPU T4 VM) ----
  # * builds CUDA kernels only if it sees a GPU or FORCE_CUDA=1; arch list needed when no GPU
  #   is visible. ONE arch (A100 = 8.0): Objects cannot run on the L4 anyway (32 GB floor) and
  #   every extra arch doubles compile time.
  # * --no-build-isolation: pip's isolated build env pulls the NEWEST numpy, compiles the
  #   extension against numpy-2 headers, then the runtime env (pinned lower by [dev]/[inference])
  #   has numpy 1.x -> "numpy.dtype size changed, expected 96 got 88" at import (measured
  #   2026-09-11, cost a 155-min build). Building against the env's own numpy avoids it.
  # * the built WHEEL is cached to Drive: any later session installs it in seconds, even if
  #   something else in the env has to be rebuilt.
  export FORCE_CUDA=1
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
  NVCC_PY=$("$OBJ_PY" -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))" 2>/dev/null || true)
  if [ -n "$NVCC_PY" ] && [ -x "$NVCC_PY/bin/nvcc" ]; then export CUDA_HOME="$NVCC_PY"; fi
  P3D_REF="git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"   # = requirements.p3d.txt (v0.7.8)
  NP_VER=$("$OBJ_PY" -c "import numpy; print(numpy.__version__)")
  say "p3d build: FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST CUDA_HOME=${CUDA_HOME:-<unset>} nvcc: $( (command -v nvcc || ls "${CUDA_HOME:-/nonexistent}/bin/nvcc") 2>/dev/null || echo NOT FOUND ); env numpy $NP_VER; MAX_JOBS=$MAX_JOBS"
  step "build deps (ninja, setuptools, wheel)"                        "$OBJ_PY" -m pip install -q ninja setuptools wheel
  P3D_WHL=$(ls "$WHEEL_DIR"/pytorch3d-*.whl 2>/dev/null | head -1 || true)
  if [ -n "$P3D_WHL" ]; then
    say "pytorch3d: cached wheel found on Drive: $(basename "$P3D_WHL")"
  else
    step "pytorch3d WHEEL build (no build isolation, ~1-2 h on 2 vCPU; cached to Drive after)"          "$OBJ_PY" -m pip wheel -q --no-build-isolation --no-deps -w "$WHEEL_DIR" "pytorch3d @ $P3D_REF"
    P3D_WHL=$(ls "$WHEEL_DIR"/pytorch3d-*.whl 2>/dev/null | head -1 || true)
  fi
  if [ -n "$P3D_WHL" ]; then
    step "pytorch3d install from wheel"                                "$OBJ_PY" -m pip install -q --no-deps "$P3D_WHL"
  else
    say "objects: pytorch3d wheel build FAILED (see $LOG)"
  fi
  # [p3d] extras are now both satisfied (pytorch3d from the wheel, flash_attn from the release
  # wheel); running `pip install -e .[p3d]` would re-resolve the git URL and REBUILD, so skip it.
  export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
  step "[inference] (kaolin 0.17.0, gsplat, gradio)"                 "$OBJ_PY" -m pip install -q -e '.[inference]'
  step "hydra patch (a PYTHON script — run with the env python, not bash)" "$OBJ_PY" ./patching/hydra
  NP_VER2=$("$OBJ_PY" -c "import numpy; print(numpy.__version__)")
  [ "$NP_VER" = "$NP_VER2" ] || say "WARNING: numpy changed during install: $NP_VER -> $NP_VER2 (compiled extensions built against $NP_VER)"
  step "probe deps (scipy, opencv-headless; numpy held at $NP_VER2)"  "$OBJ_PY" -m pip install -q "numpy==$NP_VER2" scipy opencv-python-headless
  mkdir -p "$OBJ_DIR/checkpoints"
  python - <<PY 2>&1 | tee -a "$LOG"
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
obj = Path("$OBJ_DIR")
try:
    p = Path(snapshot_download(repo_id="facebook/sam-3d-objects", repo_type="model",
                               local_dir=str(obj / "checkpoints" / "hf-download"), max_workers=1))
    src, dst = p / "checkpoints", obj / "checkpoints" / "hf"
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
    print("[setup_sam3d] objects weights ->", dst, "pipeline.yaml present:", (dst / "pipeline.yaml").exists())
except Exception as e:
    print(f"[setup_sam3d] objects weights NOT downloaded: {type(e).__name__}: {str(e)[:160]}")
PY
  # GPU-capability check that needs NO GPU: a CUDA-enabled pytorch3d links libc10_cuda /
  # libcudart; a CPU-only build does not. Refuse to cache a CPU-only env.
  # MEASURED 2026-09-12: pytorch3d._C must be imported AFTER torch (its libc10.so comes from
  # torch's loaded libs; importing _C alone fails with "libc10.so: cannot open shared object")
  # — that false negative discarded a good build. And sam3d_objects/__init__ imports an
  # internal, unshipped `sam3d_objects.init` unless LIDRA_SKIP_INIT is set, exactly as the
  # vendor's notebook/inference.py does (it also sets CUDA_HOME=$CONDA_PREFIX).
  export LIDRA_SKIP_INIT=true
  export CONDA_PREFIX="$MM_ROOT/envs/$ENV_NAME" CUDA_HOME="$MM_ROOT/envs/$ENV_NAME"
  P3D_SO=$("$OBJ_PY" -c "import torch, pytorch3d._C as C; print(C.__file__)" 2>/dev/null || true)
  P3D_CUDA=0
  if [ -n "$P3D_SO" ] && ldd "$P3D_SO" 2>/dev/null | grep -qE "c10_cuda|cudart"; then P3D_CUDA=1; fi
  say "pytorch3d CUDA build: $P3D_CUDA ($P3D_SO; import error, if any: $("$OBJ_PY" -c 'import torch, pytorch3d._C' 2>&1 | tail -1))"
  if "$OBJ_PY" -c "import torch, kaolin, pytorch3d, sam3d_objects; print('[setup_sam3d] env OK: torch', torch.__version__, 'cuda', torch.version.cuda, 'kaolin', kaolin.__version__)" 2>&1 | tee -a "$LOG" && [ "$P3D_CUDA" = 1 ]; then
    if [ ! -f "$ENV_TAR" ]; then
      say "packing env to Drive for later sessions (one-time, large)"
      mkdir -p "$(dirname "$ENV_TAR")" && tar -cf "$ENV_TAR" -C "$MM_ROOT/envs" "$ENV_NAME" && touch "$ENV_TAR.ok" && say "packed -> $ENV_TAR ($(du -h "$ENV_TAR" | cut -f1))"
    fi
    say "run the Objects probe with:  $OBJ_PY scripts/probe_sam3d.py --p0 --variant objects"
  else
    [ "$P3D_CUDA" = 1 ] || say "pytorch3d was built WITHOUT CUDA kernels — NOT cached. Rebuild on a GPU runtime (an L4 suffices for the build) or with FORCE_CUDA=1 and a working nvcc; see $LOG"
    say "objects env NOT complete — the failing step above is the P0 finding; do not work around it silently"
  fi
fi
say "done; log at $LOG"
