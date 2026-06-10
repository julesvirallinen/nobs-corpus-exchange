from transformers import pipeline

TOPIC_LABELS = [
    "immigration or refugees",
    "religion or religious groups",
    "race or ethnicity",
    "gender or women",
    "LGBTQ+ or transgender people",
    "politics or political groups",
]

CATEGORY_LABELS = [
    "threat of violence",
    "profanity or obscene language",
    "insult or personal attack",
    "identity-based hatred",
    "general toxicity",
]

CATEGORY_MAP = {
    "threat of violence": "THREAT",
    "profanity or obscene language": "PROFANITY",
    "insult or personal attack": "INSULT",
    "identity-based hatred": "IDENTITY_ATTACK",
    "general toxicity": "TOXICITY",
}

TOPIC_MAP = {label: label.split(" or ")[0] for label in TOPIC_LABELS}

# Confidence below this → mark as GENERATE_TOPIC (no confident predefined match)
TOPIC_CONFIDENCE_THRESHOLD = 0.25


class FinBERTBackend:
    def __init__(self, threshold: float = 0.6):
        self._threshold = threshold
        print("Loading xlm-roberta-large-xnli (zero-shot, multilingual)...")
        self._pipe = pipeline(
            "zero-shot-classification",
            model="joeddav/xlm-roberta-large-xnli",
        )

    def classify(self, text: str) -> dict:
        topic_result = self._pipe(text, TOPIC_LABELS, multi_label=False)
        category_result = self._pipe(text, CATEGORY_LABELS, multi_label=True)

        top_score = topic_result["scores"][0]
        if top_score >= TOPIC_CONFIDENCE_THRESHOLD:
            top_topic = TOPIC_MAP.get(topic_result["labels"][0], topic_result["labels"][0])
        else:
            # No confident predefined topic — flag for manual/LLM generation
            top_topic = "GENERATE_TOPIC"

        categories = [
            CATEGORY_MAP[label]
            for label, score in zip(category_result["labels"], category_result["scores"])
            if score >= self._threshold
        ]

        return {
            "topic": top_topic,
            "topic_score": round(top_score, 3),
            "categories": categories,
            "category_scores": {
                CATEGORY_MAP[label]: round(score, 3)
                for label, score in zip(category_result["labels"], category_result["scores"])
            },
        }
