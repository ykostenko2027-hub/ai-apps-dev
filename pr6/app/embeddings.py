import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model

def embed_passages(texts: List[str]) -> np.ndarray:
    model = get_model()
    prefixed_texts = [f"passage: {t}" for t in texts]
    vectors = model.encode(prefixed_texts, normalize_embeddings=True)
    return np.array(vectors, dtype=np.float32)

def embed_query(text: str) -> np.ndarray:
    model = get_model()
    prefixed_text = f"query: {text}"
    vector = model.encode(prefixed_text, normalize_embeddings=True)
    return np.array(vector, dtype=np.float32)