"""Generative paraphrase proposers (mock + llama.cpp + Unsloth Qwen3.5)."""

from __future__ import annotations

from typing import List, Optional, Protocol, Sequence

import numpy as np

from .config import EmHsdConfig
from .qwen_models import (
    is_gguf_repo,
    llama_server_model_id,
    resolve_unsloth_safetensors_model,
)

_PARAPHRASE_PROMPT = """Rewrite the post below with different wording and style.
Rules:
- KEEP these terms unchanged in meaning: {protected_list}
- Do not remove insults or soften offensive content
- Change distinctive phrasing, openers, and stylistic tics
- Similar length (±25%)
Output only the rewritten post.

Post: {text}"""


def _tokenize_prompt(tokenizer, formatted: str, device):
    """Tokenize chat text; Qwen3-VL processors treat positional args as images."""
    import torch

    try:
        inputs = tokenizer(text=formatted, return_tensors="pt")
    except TypeError:
        inputs = tokenizer(formatted, return_tensors="pt")
    return inputs.to(device)


def _cuda_usable() -> bool:
    """Return True only if CUDA kernels can actually run (GTX 1060 / sm_61 may fail)."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        torch.zeros(1, device="cuda")
        return True
    except Exception:
        return False


class GenerativeProposer(Protocol):
    def bind(self, rng: np.random.Generator, protected_terms: Sequence[str]) -> None:
        ...

    def propose(self, text: str, k: int) -> List[str]:
        ...


class MockProposer:
    """Deterministic paraphrase variants for unit tests (no GPU)."""

    _OPENERS = (
        "Honestly,", "Look,", "For what it's worth,", "I mean,", "Seriously,",
        "To be fair,", "Not gonna lie,", "Real talk,",
    )
    _CLOSERS = (
        " anyway.", " if you ask me.", " just saying.", " period.",
        " no cap.", " for real.", " imo.",
    )

    def __init__(self, config: EmHsdConfig):
        self.config = config
        self._rng: Optional[np.random.Generator] = None
        self._protected: List[str] = []

    def bind(self, rng: np.random.Generator, protected_terms: Sequence[str]) -> None:
        self._rng = rng
        self._protected = list(protected_terms)

    def propose(self, text: str, k: int) -> List[str]:
        if self._rng is None:
            raise RuntimeError("MockProposer.bind() must be called before propose()")
        rng = self._rng
        out: List[str] = []
        seen = set()
        for i in range(max(k * 2, k)):
            opener = self._OPENERS[(i + int(rng.integers(1000))) % len(self._OPENERS)]
            closer = self._CLOSERS[int(rng.integers(len(self._CLOSERS)))]
            body = text.strip()
            if body.endswith("."):
                body = body[:-1]
            cand = f"{opener} {body}{closer}"
            if cand not in seen:
                seen.add(cand)
                out.append(cand)
            if len(out) >= k:
                break
        return out[:k]


class TransformersQwenProposer:
    """Qwen paraphrase via HuggingFace transformers (CPU/GPU fallback when Unsloth fails)."""

    def __init__(self, config: EmHsdConfig):
        self.config = config
        self._rng: Optional[np.random.Generator] = None
        self._protected: List[str] = []
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gen = self.config.generation
        model_name = resolve_unsloth_safetensors_model(
            gen.model, gen.unsloth_model,
        )
        if is_gguf_repo(gen.model) and model_name == gen.model:
            raise ValueError(
                f"generation.model {gen.model!r} is a GGUF repo; set "
                "generation.unsloth_model to a safetensors id (e.g. Qwen/Qwen3.5-0.8B) "
                "or use generation.backend: llama_cpp with llama-server."
            )
        print(f"Loading transformers model: {model_name}", flush=True)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        use_cuda = _cuda_usable()
        try:
            if use_cuda and gen.load_in_4bit:
                from transformers import BitsAndBytesConfig
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                dtype = torch.float32
                device = "cpu" if not use_cuda else "cuda"
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=dtype,
                    trust_remote_code=True,
                ).to(device)
        except Exception as exc:
            print(f"WARN: quantized load failed ({exc}); using CPU float32.", flush=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            ).to("cpu")
        self._model.eval()

    def bind(self, rng: np.random.Generator, protected_terms: Sequence[str]) -> None:
        self._rng = rng
        self._protected = list(protected_terms)

    def propose(self, text: str, k: int) -> List[str]:
        if self._rng is None:
            raise RuntimeError("TransformersQwenProposer.bind() must be called before propose()")
        self._load()
        gen = self.config.generation
        protected_list = ", ".join(self._protected) if self._protected else "(none)"
        prompt = _PARAPHRASE_PROMPT.format(protected_list=protected_list, text=text)
        messages = [{"role": "user", "content": prompt}]
        tokenizer = self._tokenizer
        model = self._model

        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_template:
            formatted = apply_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        import torch

        device = next(model.parameters()).device
        out: List[str] = []
        seen = set()
        attempts = 0
        max_attempts = k * 3
        while len(out) < k and attempts < max_attempts:
            attempts += 1
            seed = int(self._rng.integers(2**31))
            torch.manual_seed(seed)
            inputs = _tokenize_prompt(tokenizer, formatted, device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=gen.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.em_hsd_v2.generation_temperature,
                    top_p=gen.top_p,
                    top_k=gen.top_k,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            cand = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out


class UnslothQwenProposer:
    """Local Qwen3.5 paraphrase via Unsloth FastLanguageModel."""

    def __init__(self, config: EmHsdConfig):
        self.config = config
        self._rng: Optional[np.random.Generator] = None
        self._protected: List[str] = []
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import os
        # GTX 10xx (sm_61): Triton/inductor compile fails at generation time.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        import torch
        from unsloth import FastLanguageModel

        gen = self.config.generation
        model_name = resolve_unsloth_safetensors_model(
            gen.model, gen.unsloth_model,
        )
        if is_gguf_repo(gen.model) and not gen.unsloth_model.strip():
            model_name = resolve_unsloth_safetensors_model(gen.model)
        use_4bit = gen.load_in_4bit and _cuda_usable()
        if gen.load_in_4bit and not _cuda_usable():
            print(
                "WARN: CUDA unavailable — loading Qwen in bf16/fp16 on CPU (slow).",
                flush=True,
            )
        kwargs = dict(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=use_4bit,
            fast_inference=False,
        )
        if not use_4bit:
            kwargs["load_in_4bit"] = False
            kwargs["dtype"] = None
        try:
            self._model, self._tokenizer = FastLanguageModel.from_pretrained(**kwargs)
        except Exception as exc:
            if use_4bit:
                print(f"WARN: 4-bit load failed ({exc}); retrying without 4-bit.", flush=True)
                kwargs["load_in_4bit"] = False
                self._model, self._tokenizer = FastLanguageModel.from_pretrained(**kwargs)
            else:
                raise
        FastLanguageModel.for_inference(self._model)

    def bind(self, rng: np.random.Generator, protected_terms: Sequence[str]) -> None:
        self._rng = rng
        self._protected = list(protected_terms)

    def propose(self, text: str, k: int) -> List[str]:
        if self._rng is None:
            raise RuntimeError("UnslothQwenProposer.bind() must be called before propose()")
        self._load()
        gen = self.config.generation
        protected_list = ", ".join(self._protected) if self._protected else "(none)"
        prompt = _PARAPHRASE_PROMPT.format(protected_list=protected_list, text=text)
        messages = [{"role": "user", "content": prompt}]
        tokenizer = self._tokenizer
        model = self._model

        apply_template = getattr(tokenizer, "apply_chat_template", None)
        if apply_template:
            formatted = apply_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            formatted = prompt

        import torch

        out: List[str] = []
        seen = set()
        attempts = 0
        max_attempts = k * 3
        while len(out) < k and attempts < max_attempts:
            attempts += 1
            seed = int(self._rng.integers(2**31))
            torch.manual_seed(seed)
            inputs = _tokenize_prompt(tokenizer, formatted, model.device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=gen.max_new_tokens,
                    do_sample=True,
                    temperature=self.config.em_hsd_v2.generation_temperature,
                    top_p=gen.top_p,
                    top_k=gen.top_k,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = generated[0][inputs["input_ids"].shape[1]:]
            cand = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out


class LlamaServerProposer:
    """Paraphrase via llama-server OpenAI API (Unsloth GGUF path).

    Start the server first, e.g. ``scripts/start_llama_server.sh``.
    See ``resources/qwen-run-loclay.md`` § EM-HSD integration.
    """

    def __init__(self, config: EmHsdConfig):
        self.config = config
        self._rng: Optional[np.random.Generator] = None
        self._protected: List[str] = []
        self._client = None
        self._model_id: str = ""

    def _load(self):
        if self._client is not None:
            return
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "llama_cpp backend requires openai: pip install openai"
            ) from exc

        gen = self.config.generation
        self._client = OpenAI(
            base_url=gen.llama_server_url,
            api_key=gen.llama_server_api_key,
        )
        self._model_id = llama_server_model_id(gen.model, gen.llama_model_alias)
        try:
            models = self._client.models.list()
            ids = [m.id for m in models.data]
            if ids and self._model_id not in ids:
                print(
                    f"WARN: llama-server model {self._model_id!r} not in {ids}; "
                    f"using first available: {ids[0]!r}",
                    flush=True,
                )
                self._model_id = ids[0]
        except Exception as exc:
            raise RuntimeError(
                f"llama-server not reachable at {gen.llama_server_url}: {exc}. "
                "Start it with: em-hsd-v2/scripts/start_llama_server.sh"
            ) from exc

    def bind(self, rng: np.random.Generator, protected_terms: Sequence[str]) -> None:
        self._rng = rng
        self._protected = list(protected_terms)

    def propose(self, text: str, k: int) -> List[str]:
        if self._rng is None:
            raise RuntimeError("LlamaServerProposer.bind() must be called before propose()")
        self._load()
        gen = self.config.generation
        em = self.config.em_hsd_v2
        protected_list = ", ".join(self._protected) if self._protected else "(none)"
        prompt = _PARAPHRASE_PROMPT.format(protected_list=protected_list, text=text)
        messages = [{"role": "user", "content": prompt}]

        extra_body = {
            "top_k": gen.top_k,
            "min_p": gen.min_p,
            "repetition_penalty": gen.repetition_penalty,
        }
        if gen.enable_thinking:
            extra_body["chat_template_kwargs"] = {"enable_thinking": True}

        out: List[str] = []
        seen = set()
        attempts = 0
        max_attempts = k * 3
        while len(out) < k and attempts < max_attempts:
            attempts += 1
            try:
                response = self._client.chat.completions.create(
                    model=self._model_id,
                    messages=messages,
                    temperature=em.generation_temperature,
                    top_p=gen.top_p,
                    presence_penalty=gen.presence_penalty,
                    max_tokens=gen.max_new_tokens,
                    extra_body=extra_body,
                )
            except Exception as exc:
                raise RuntimeError(f"llama-server completion failed: {exc}") from exc
            cand = (response.choices[0].message.content or "").strip()
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
        return out


def make_proposer(config: EmHsdConfig) -> GenerativeProposer:
    backend = config.generation.backend
    if backend == "llama_cpp":
        if not is_gguf_repo(config.generation.model):
            print(
                f"WARN: generation.backend=llama_cpp but model "
                f"{config.generation.model!r} is not a *-GGUF repo.",
                flush=True,
            )
        return LlamaServerProposer(config)
    if backend == "unsloth":
        if is_gguf_repo(config.generation.model) and not config.generation.unsloth_model.strip():
            print(
                f"WARN: loading safetensors {resolve_unsloth_safetensors_model(config.generation.model)!r} "
                f"for GGUF config {config.generation.model!r}; "
                "prefer backend: llama_cpp for GGUF per Unsloth docs.",
                flush=True,
            )
        return UnslothQwenProposer(config)
    if backend == "transformers":
        return TransformersQwenProposer(config)
    if backend == "mock":
        return MockProposer(config)
    raise ValueError(
        f"unknown generation.backend {backend!r} "
        "(expected mock|llama_cpp|unsloth|transformers)"
    )


def get_proposer(config: EmHsdConfig) -> GenerativeProposer:
    cached = getattr(config, "_proposer", None)
    if cached is not None:
        return cached
    proposer = make_proposer(config)
    config._proposer = proposer
    return proposer
