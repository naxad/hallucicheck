# verifier/services/nli.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"

tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
model.eval()

@torch.inference_mode()
def nli_scores(premise: str, hypothesis: str) -> dict:
    inputs = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    logits = model(**inputs).logits[0]  # shape: (num_labels,)
    probs = torch.softmax(logits, dim=-1).tolist()

    # ✅ map based on id2label, not fixed indices
    out = {"entailment": 0.0, "neutral": 0.0, "contradiction": 0.0}
    id2label = model.config.id2label  # e.g. {0:"CONTRADICTION",1:"NEUTRAL",2:"ENTAILMENT"}

    for i, p in enumerate(probs):
        label = str(id2label.get(i, "")).lower()
        if "entail" in label:
            out["entailment"] = float(p)
        elif "neutral" in label:
            out["neutral"] = float(p)
        elif "contrad" in label:
            out["contradiction"] = float(p)

    return out