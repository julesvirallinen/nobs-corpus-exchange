#!/usr/bin/env bash
# Create conda env for TRIAGE-DP Layer 4 (EM-HSD v2 + Unsloth Qwen).
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

echo "==> Pre-download Qwen3.5-0.8B"
python - <<'PY'
import torch
from unsloth import FastLanguageModel
use_4bit = False
if torch.cuda.is_available():
    torch.zeros(1, device="cuda")
    use_4bit = True
model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3.5-0.8B",
    max_seq_length=2048,
    load_in_4bit=use_4bit,
    fast_inference=False,
)
FastLanguageModel.for_inference(model)
out = tok(text="Hello", return_tensors="pt")
print("Qwen3.5-0.8B loaded OK (4bit=" + str(use_4bit) + ", tokens=" + str(out["input_ids"].shape) + ")")
PY

echo "==> Re-pin torch cu118 (unsloth may upgrade torch to cu128)"
pip install \
  "torch==2.6.0+cu118" \
  "torchvision==0.21.0+cu118" \
  "torchaudio==2.6.0+cu118" \
  --extra-index-url https://download.pytorch.org/whl/cu118 --force-reinstall

echo ""
echo "Done. Activate with:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "SPINE + runtime:"
echo "  python ${NOBS}/Johnny\\ t0-1.03/scripts/setup_lexicons.py"
echo "  export PYTHONPATH=\"src:${NOBS}/Johnny t0-1.03/src\""
echo "  export TORCHDYNAMO_DISABLE=1"
echo "  export EM_HSD_ALLOW_DOWNLOADS=1"
echo ""
echo "Smoke test:"
echo "  cd ${ROOT} && PYTHONPATH=src python scripts/test_qwen.py --config configs/em-hsd-v2-qwen35-08b.yaml"
