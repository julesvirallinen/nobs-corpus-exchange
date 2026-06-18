#!/usr/bin/env bash
# Create conda env for TRIAGE-DP Layer 4 (llama.cpp Qwen GGUF + HF stack).
set -euo pipefail

ENV_NAME="${ENV_NAME:-em-hsd-qwen}"
PYTHON_VER="${PYTHON_VER:-3.10}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOBS="$(cd "${ROOT}/.." && pwd)"

echo "==> Creating conda env: ${ENV_NAME} (python ${PYTHON_VER})"
conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "==> Layer 4 Python deps (see ${NOBS}/requirements-layer-4.txt)"
pip install --upgrade pip
pip install -r "${NOBS}/requirements-layer-4.txt" \
  --extra-index-url https://download.pytorch.org/whl/cu118

echo "==> EM-HSD v2 package"
pip install -e "${ROOT}"

echo "==> Download Qwen3.5-0.8B-GGUF (llama.cpp)"
python "${ROOT}/scripts/download_qwen_gguf.py" || true

echo ""
echo "Build llama.cpp (needs cmake + build-essential):"
echo "  bash ${ROOT}/scripts/build_llama_cpp.sh"
echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "SPINE + runtime:"
echo "  python ${NOBS}/Johnny\\ t0-1.03/scripts/setup_lexicons.py"
echo "  export PYTHONPATH=\"src:${NOBS}/Johnny t0-1.03/src\""
echo "  export EM_HSD_ALLOW_DOWNLOADS=1"
echo ""
echo "1) Start llama-server (terminal 1):"
echo "  cd ${ROOT} && bash scripts/start_llama_server.sh"
echo ""
echo "2) Smoke test (terminal 2):"
echo "  cd ${ROOT} && PYTHONPATH=src python scripts/test_qwen.py --config configs/em-hsd-v2.yaml"
