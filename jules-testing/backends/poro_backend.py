import json
import subprocess
import sys
from pathlib import Path
from mlx_lm import load, generate

HF_MODEL = "Finnish-NLP/llama-7b-finnish-instruct-v0.2"
LOCAL_MODEL = Path(__file__).parent.parent / "models" / "finnish-7b-4bit"

INSTRUCTION_TEMPLATE = (
    'Classify Finnish text for hate speech. Output ONLY a JSON object.\n'
    '"topic": immigration|religion|race_or_ethnicity|gender_or_women|lgbtq|politics or short English phrase\n'
    '"categories": array from [THREAT, PROFANITY, INSULT, IDENTITY_ATTACK, TOXICITY]\n\n'
    'Example:\n'
    'Text: "Naiset eivät kuulu johtotehtäviin"\n'
    '{"topic": "gender_or_women", "categories": ["IDENTITY_ATTACK"]}\n\n'
    "Text: "
)


def _ensure_model():
    if LOCAL_MODEL.exists():
        return
    print(f"Converting {HF_MODEL} to 4-bit MLX format (one-time, ~13GB download)...")
    LOCAL_MODEL.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "mlx_lm.convert",
            "--hf-path", HF_MODEL,
            "--mlx-path", str(LOCAL_MODEL),
            "--quantize",
            "--q-bits", "4",
        ],
        check=True,
    )


class PoroBackend:
    def __init__(self, threshold: float = 0.6):
        _ensure_model()
        print(f"Loading Finnish 7B 4-bit from {LOCAL_MODEL} ...")
        self._model, self._tokenizer = load(str(LOCAL_MODEL))

    def classify(self, text: str) -> dict:
        # Pre-fill response with `{` to force JSON output
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{INSTRUCTION_TEMPLATE}{text}\n\n"
            '### Response:\n{"topic": "'
        )
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=80,
            verbose=False,
        )
        # Re-attach the pre-filled opening
        raw = ('{"topic": "' + response).strip()

        try:
            start = raw.index("{")
            end = raw.rindex("}") + 1
            data = json.loads(raw[start:end])
            return {
                "topic": data.get("topic", "GENERATE_TOPIC"),
                "topic_score": None,
                "categories": data.get("categories", []),
                "category_scores": {},
            }
        except (ValueError, json.JSONDecodeError):
            return {
                "topic": "parse_error",
                "topic_score": None,
                "categories": [],
                "category_scores": {},
                "raw": raw,
            }
