from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL = None


def get_embedding_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        # small + strong baseline model
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _MODEL


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    emb = model.encode(texts, normalize_embeddings=True)
    return np.array(emb, dtype="float32")