#!/usr/bin/env bash
# Start llama-server for Unsloth Qwen3.5-0.8B-GGUF (EM-HSD v2 paraphrase backend).
# Uses local cached GGUF files (avoids HTTPS / OpenSSL 3 requirement of -hf).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_CPP="${LLAMA_CPP:-${ROOT}/llama.cpp}"

MODEL_REPO="${QWEN_GGUF_REPO:-unsloth/Qwen3.5-0.8B-GGUF}"
QUANT="${QWEN_GGUF_QUANT:-UD-Q4_K_XL}"
PORT="${LLAMA_SERVER_PORT:-8001}"
ALIAS="${LLAMA_MODEL_ALIAS:-${MODEL_REPO}}"

TEMP="${LLAMA_TEMP:-0.7}"
TOP_P="${LLAMA_TOP_P:-0.8}"
TOP_K="${LLAMA_TOP_K:-20}"
MIN_P="${LLAMA_MIN_P:-0.00}"

if [[ ! -x "${LLAMA_CPP}/llama-server" ]]; then
  echo "ERROR: llama-server not found at ${LLAMA_CPP}/llama-server" >&2
  echo "Build it: bash scripts/build_llama_cpp.sh" >&2
  exit 1
fi

mapfile -t GGUF_PATHS < <(python "${ROOT}/scripts/resolve_qwen_gguf_paths.py")
if [[ ${#GGUF_PATHS[@]} -lt 2 ]]; then
  echo "ERROR: could not resolve GGUF paths; run: python scripts/download_qwen_gguf.py" >&2
  exit 1
fi
MODEL_FILE="${GGUF_PATHS[0]}"
MMPROJ_FILE="${GGUF_PATHS[1]}"

EXTRA=(--chat-template-kwargs '{"enable_thinking":false}')
if [[ "${QWEN_ENABLE_THINKING:-false}" == "true" ]]; then
  EXTRA=(--chat-template-kwargs '{"enable_thinking":true}')
fi

echo "Starting llama-server (local GGUF)"
echo "  model:  ${MODEL_FILE}"
echo "  mmproj: ${MMPROJ_FILE}"
echo "  port:   ${PORT}"

exec "${LLAMA_CPP}/llama-server" \
  --model "${MODEL_FILE}" \
  --mmproj "${MMPROJ_FILE}" \
  --host 127.0.0.1 \
  --temp "${TEMP}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --min-p "${MIN_P}" \
  --alias "${ALIAS}" \
  --port "${PORT}" \
  "${EXTRA[@]}"
