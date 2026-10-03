import os
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from .documents import Chunk
from .embeddings import embed_query

INDEX_DIR = "index"


@dataclass
class BuiltIndex:
    chunks: List[Chunk]
    vectors: np.ndarray
    model_name: str


def build(chunks: List[Chunk], vectors: np.ndarray, model_name: str) -> BuiltIndex:
    return BuiltIndex(chunks=chunks, vectors=vectors, model_name=model_name)


def save(built: BuiltIndex, index_dir: str = INDEX_DIR) -> None:
    os.makedirs(index_dir, exist_ok=True)
    np.save(os.path.join(index_dir, "vectors.npy"), built.vectors)

    meta = []
    for c in built.chunks:
        meta.append({
            "id": c.id,
            "text": c.text,
            "source": c.source,
            "metadata": c.metadata
        })
    with open(os.path.join(index_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_name": built.model_name,
            "chunks": meta
        }, f, ensure_ascii=False, indent=2)


def load_index(index_dir: str = INDEX_DIR):
    v_path = os.path.join(index_dir, "vectors.npy")
    m_path = os.path.join(index_dir, "metadata.json")
    if not os.path.exists(v_path) or not os.path.exists(m_path):
        return None, None
    vectors = np.load(v_path)
    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("chunks", data)
    return vectors, meta


def semantic_search(query: str, top_k: int = 8) -> List[Dict[str, Any]]:
    vectors, metadata = load_index()
    if vectors is None or metadata is None:
        return []

    query_vector = embed_query(query)
    scores = np.dot(vectors, query_vector)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        item = dict(metadata[idx])
        item["score"] = float(scores[idx])
        results.append(item)
    return results