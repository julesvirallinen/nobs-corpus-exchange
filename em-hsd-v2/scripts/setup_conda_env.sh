#!/usr/bin/env bash
# Create conda env `em-hsd-qwen` with Unsloth + Qwen3.5-0.8B deps for EM-HSD v2.
# Target: NVIDIA GTX 1060 (CUDA 11.8 / sm_61).
set -euo pipefail

ENV_NAME="${ENV_NAME:-em-hsd-qwen}"
PYTHON_VER="${PYTHON_VER:-3.11}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NOBS="$(cd "$ROOT/.." && pwd)"

echo "==> Creating conda env: ${ENV_NAME} (python ${PYTHON_VER})"
conda create -n "${ENV_NAME}" "python=${PYTHON_VER}" -y

# shellcheck disable=SC1091
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "==> PyTorch (CUDA 11.8 — GTX 10xx compatible)"
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "==> Unsloth + HF stack"
pip install "unsloth" transformers accelerate datasets trl peft bitsandbytes
pip install sentence-transformers safetensors huggingface_hub

echo "==> EM-HSD v2 + SPINE core deps"
pip install numpy scikit-learn pyyaml regex pytest
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
import unsloth
print("unsloth ok")
PY

echo "==> Pre-download Qwen3.5-0.8B (optional cache warm-up)"
python - <<'PY'
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-0.8B",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=False,
)
FastLanguageModel.for_inference(model)
print("Qwen3.5-0.8B loaded OK")
PY

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Smoke test:"
echo "  cd ${ROOT} && PYTHONPATH=src python scripts/test_qwen.py --config configs/em-hsd-v2-qwen35-08b.yaml"
