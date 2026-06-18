"""Qwen model id helpers (Unsloth GGUF vs safetensors)."""

from __future__ import annotations

# Unsloth "How to Run Locally": GGUF → llama.cpp; safetensors → FastLanguageModel.
_GGUF_TO_SAFETENSORS = {
    "unsloth/Qwen3.5-0.8B-GGUF": "Qwen/Qwen3.5-0.8B",
    "unsloth/Qwen3.5-2B-GGUF": "unsloth/Qwen3.5-2B",
    "unsloth/Qwen3.5-4B-GGUF": "unsloth/Qwen3.5-4B",
    "unsloth/Qwen3.5-9B-GGUF": "unsloth/Qwen3.5-9B",
}


def is_gguf_repo(model: str) -> bool:
    return model.rstrip("/").endswith("-GGUF")


def resolve_unsloth_safetensors_model(model: str, override: str = "") -> str:
    """Map a GGUF repo id to a HuggingFace safetensors model for FastLanguageModel."""
    if override.strip():
        return override.strip()
    if model in _GGUF_TO_SAFETENSORS:
        return _GGUF_TO_SAFETENSORS[model]
    if is_gguf_repo(model):
        return model[:-5]
    return model


def llama_server_model_id(model: str, alias: str = "") -> str:
    """OpenAI API model name served by llama-server (--alias)."""
    if alias.strip():
        return alias.strip()
    return model
