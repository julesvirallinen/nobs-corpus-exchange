#!/usr/bin/env bash
# Start llama-server for Unsloth Qwen3.5-0.8B-GGUF (EM-HSD v2 paraphrase backend).
# Aligns with resources/qwen-run-loclay.md — Qwen3.5 Small, non-thinking instruct.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_CPP="${LLAMA_CPP:-${ROOT}/llama.cpp}"

MODEL_REPO="${QWEN_GGUF_REPO:-unsloth/Qwen3.5-0.8B-GGUF}"
QUANT="${QWEN_GGUF_QUANT:-UD-Q4_K_XL}"
PORT="${LLAMA_SERVER_PORT:-8001}"
ALIAS="${LLAMA_MODEL_ALIAS:-${MODEL_REPO}}"

# Instruct (non-thinking) general tasks — Unsloth doc §86–115
TEMP="${LLAMA_TEMP:-0.7}"
TOP_P="${LLAMA_TOP_P:-0.8}"
TOP_K="${LLAMA_TOP_K:-20}"
MIN_P="${LLAMA_MIN_P:-0.00}"

export LLAMA_CACHE="${LLAMA_CACHE:-${MODEL_REPO}}"

if [[ ! -x "${LLAMA_CPP}/llama-server" ]]; then
  echo "ERROR: llama-server not found at ${LLAMA_CPP}/llama-server" >&2
  echo "Build llama.cpp per resources/qwen-run-loclay.md (Qwen3.5 Small section)." >&2
  exit 1
fi

EXTRA=()
if [[ "${QWEN_ENABLE_THINKING:-false}" == "true" ]]; then
  EXTRA+=(--chat-template-kwargs '{"enable_thinking":true}')
fi

echo "Starting llama-server: ${MODEL_REPO}:${QUANT} on port ${PORT}"
exec "${LLAMA_CPP}/llama-server" \
  -hf "${MODEL_REPO}:${QUANT}" \
  --temp "${TEMP}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --min-p "${MIN_P}" \
  --alias "${ALIAS}" \
  --port "${PORT}" \
  "${EXTRA[@]}"
