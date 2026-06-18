#!/usr/bin/env bash
# Build llama.cpp for EM-HSD v2 (Unsloth Qwen3.5 GGUF / llama-server backend).
# Follows resources/qwen-run-loclay.md — Qwen3.5 Small, step 1.
#
# Usage:
#   bash scripts/build_llama_cpp.sh              # auto CUDA if available
#   GGML_CUDA=OFF bash scripts/build_llama_cpp.sh   # CPU only
#   JOBS=4 bash scripts/build_llama_cpp.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_CPP_DIR:-${ROOT}/llama.cpp}"
LLAMA_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp.git}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

use_cuda() {
  if [[ "${GGML_CUDA:-auto}" == "OFF" ]]; then
    return 1
  fi
  if [[ "${GGML_CUDA:-auto}" == "ON" ]]; then
    return 0
  fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

need_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing required command: $1" >&2
    echo "Install build deps (Ubuntu/Debian):" >&2
    echo "  sudo apt-get install -y build-essential cmake git curl libcurl4-openssl-dev pciutils" >&2
    exit 1
  fi
}

need_cmd git

cmake_bin() {
  if command -v cmake >/dev/null 2>&1; then
    command -v cmake
    return
  fi
  echo "ERROR: cmake not found" >&2
  exit 1
}

cmake_version_ok() {
  local ver
  ver="$(cmake --version 2>/dev/null | awk '/version/{print $3}')"
  [[ -n "${ver}" ]] || return 1
  local major minor
  major="${ver%%.*}"
  minor="${ver#*.}"; minor="${minor%%.*}"
  [[ "${major}" -gt 3 ]] || { [[ "${major}" -eq 3 ]] && [[ "${minor}" -ge 18 ]]; }
}

ensure_cmake() {
  if cmake_version_ok; then
    return
  fi
  echo "WARN: cmake $(cmake --version 2>/dev/null | awk '/version/{print $3}') < 3.18 (llama-server UI step needs 3.18+)"
  if [[ -n "${CONDA_PREFIX:-}" ]] && command -v conda >/dev/null 2>&1; then
    echo "==> Installing cmake>=3.18 into active conda env: ${CONDA_DEFAULT_ENV:-unknown}"
    conda install -y -c conda-forge "cmake>=3.18"
    hash -r
  fi
  if ! cmake_version_ok; then
    echo "ERROR: need cmake >= 3.18. Try: conda install -c conda-forge cmake" >&2
    exit 1
  fi
}

ensure_cmake
CMAKE="$(cmake_bin)"

if [[ ! -d "${LLAMA_DIR}/.git" ]]; then
  echo "==> Cloning llama.cpp into ${LLAMA_DIR}"
  git clone --depth 1 "${LLAMA_REPO}" "${LLAMA_DIR}"
else
  echo "==> Updating existing clone: ${LLAMA_DIR}"
  git -C "${LLAMA_DIR}" pull --ff-only || true
fi

CMAKE_ARGS=(
  -B "${LLAMA_DIR}/build"
  -DBUILD_SHARED_LIBS=OFF
  -DLLAMA_OPENSSL=ON
)

if use_cuda; then
  echo "==> Configuring with GGML_CUDA=ON"
  CMAKE_ARGS+=(-DGGML_CUDA=ON)
else
  echo "==> Configuring with GGML_CUDA=OFF (CPU)"
  CMAKE_ARGS+=(-DGGML_CUDA=OFF)
fi

cmake "${LLAMA_DIR}" "${CMAKE_ARGS[@]}"

echo "==> Building targets (jobs=${JOBS})"
cmake --build "${LLAMA_DIR}/build" --config Release -j "${JOBS}" --clean-first \
  --target llama-cli llama-mtmd-cli llama-server llama-gguf-split

echo "==> Installing binaries to ${LLAMA_DIR}/"
cp -f "${LLAMA_DIR}"/build/bin/llama-* "${LLAMA_DIR}/"

if [[ ! -x "${LLAMA_DIR}/llama-server" ]]; then
  echo "ERROR: build finished but llama-server missing" >&2
  exit 1
fi

echo ""
echo "OK: ${LLAMA_DIR}/llama-server"
"${LLAMA_DIR}/llama-server" --version 2>/dev/null || "${LLAMA_DIR}/llama-server" -h 2>&1 | head -3
echo ""
echo "Next:"
echo "  python scripts/download_qwen_gguf.py"
echo "  bash scripts/start_llama_server.sh"
