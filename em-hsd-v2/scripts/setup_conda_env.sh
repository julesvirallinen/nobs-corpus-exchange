#!/usr/bin/env bash
# Create conda env for EM-HSD v2 + Unsloth Qwen3.5-0.8B.
# Pins mirror the working `lum` env, with torch 2.6+cu118 for GTX 1060 (sm_61).
set -euo pipefail

ENV_NAME="${ENV_NAME:-em-hsd-qwen}"
PYTHON_VER="${PYTHON_VER:-3.10}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> Creating conda env: ${ENV_NAME} (python ${PYTHON_VER})"
conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "==> PyTorch 2.6 + CUDA 11.8 (GTX 10xx / sm_61; has torch.int1 for transformers 5.5)"
pip install --upgrade pip
pip install \
  "torch==2.6.0+cu118" \
  "torchvision==0.21.0+cu118" \
  "torchaudio==2.6.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118

echo "==> Unsloth + HF stack (from lum pins; skip xformers — needs torch>=2.10)"
pip install \
  "unsloth==2026.4.6" \
  "transformers==5.5.0" \
  "accelerate==1.12.0" \
  "datasets==4.3.0" \
  "trl==0.24.0" \
  "peft==0.18.1" \
  "bitsandbytes==0.49.1"
pip install sentence-transformers safetensors huggingface_hub regex pyyaml numpy scikit-learn pytest openai

echo "==> Re-pin torch cu118 (unsloth may upgrade torch to cu128)"
pip install \
  "torch==2.6.0+cu118" \
  "torchvision==0.21.0+cu118" \
  "torchaudio==2.6.0+cu118" \
  --index-url https://download.pytorch.org/whl/cu118 --force-reinstall

echo "==> EM-HSD v2 package"
pip install -e "${ROOT}"

echo "==> Verify CUDA + Unsloth"
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    x = torch.zeros(1, device="cuda")
    print("cuda tensor ok", x.device)
import unsloth  # noqa: F401
print("unsloth ok")
PY

echo "==> Download Qwen3.5-0.8B-GGUF (llama.cpp; see resources/qwen-run-loclay.md)"
python "${ROOT}/scripts/download_qwen_gguf.py" || true

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "1) Start llama-server (terminal 1):"
echo "  cd ${ROOT} && bash scripts/start_llama_server.sh"
echo ""
echo "2) Smoke test (terminal 2):"
echo "  cd ${ROOT} && pip install openai && PYTHONPATH=src python scripts/test_qwen.py --config configs/em-hsd-v2-qwen35-08b.yaml"
