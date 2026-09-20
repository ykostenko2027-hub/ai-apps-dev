"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих.

Індекс — це вектори всіх фрагментів, самі фрагменти з метаданими й назва
моделі, якою вектори отримано. Він будується окремою командою
(`ingest.py`) і зберігається на диску, а застосунок при старті лише
читає його: перераховувати ембедінги колекції на кожен запуск — марна
трата часу, а на кожен запит — тим паче.

Веб-рівень (`app/main.py`) звертається сюди з вектором запиту й отримує
список влучень із оцінкою схожості. Звідки береться вектор — не справа
індексу; що показувати клієнтові — не його справа теж.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яку міру схожості взяти й що зробити з векторами перед порівнянням;
* як шукати: перебір усіх векторів (для сотень фрагментів — цілком
  доречно) чи бібліотека наближеного пошуку;
* у якому вигляді зберігати індекс на диску і що обовʼязково покласти
  поруч із векторами, щоб потім не переплутати, чиї вони;
* де застосовувати фільтри за метаданими — до ранжування чи після — і що
  станеться з top-k у кожному з варіантів;
* що робити з кількома фрагментами одного документа у видачі;
* чи потрібен поріг схожості й звідки взяти його значення.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"

# Скільки результатів повертати за замовчуванням і чи відсікати слабкі.
# Порожній поріг означає «без порога».
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "5"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті.

    `vectors` — масив (кількість фрагментів × розмірність); рядок i
    відповідає `chunks[i]`. `model_name` — модель, якою отримано вектори:
    без неї індекс, збудований однією моделлю, мовчки шукатиме векторами
    іншої.
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів."""
    if len(chunks) != len(vectors):
        raise ValueError(
            f"Кількість фрагментів ({len(chunks)}) не відповідає кількості векторів ({len(vectors)})"
        )
    # L2-нормалізація векторів для косинусної схожості через скалярний добуток
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized_vectors = np.asarray(vectors / norms, dtype=np.float32)
    return SearchIndex(chunks=chunks, vectors=normalized_vectors, model_name=model_name)


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск: вектори в npy, фрагменти та назву моделі в JSON."""
    import json

    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", index.vectors)

    chunks_data = [
        {"text": c.text, "source": c.source, "metadata": c.metadata}
        for c in index.chunks
    ]
    with open(path / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    meta = {
        "model_name": index.model_name,
        "num_chunks": len(index.chunks),
        "dim": int(index.vectors.shape[1]) if len(index.chunks) > 0 else 0,
        **index.extra,
    }
    with open(path / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load(path: Path = INDEX_DIR) -> SearchIndex:
    """Прочитати індекс із диска."""
    import json

    vec_file = path / "vectors.npy"
    chunks_file = path / "chunks.json"
    meta_file = path / "meta.json"

    if not (vec_file.exists() and chunks_file.exists() and meta_file.exists()):
        raise FileNotFoundError(
            f"Індекс не знайдено у {path}. Спочатку виконайте python ingest.py"
        )

    vectors = np.load(vec_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [
        Chunk(text=c["text"], source=c["source"], metadata=c.get("metadata", {}))
        for c in chunks_data
    ]

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)

    model_name = meta.get("model_name", "unknown")
    return SearchIndex(chunks=chunks, vectors=vectors, model_name=model_name, extra=meta)


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти фрагменти, найближчі до вектора запиту.

    Застосовує pre-filtering (фільтрацію до ранжування) за метаданими.
    """
    if not index.chunks or len(index.vectors) == 0:
        return []

    # Фільтрація до ранжування (pre-filtering)
    if filters:
        valid_indices = [
            i
            for i, chunk in enumerate(index.chunks)
            if all(chunk.metadata.get(k) == v for k, v in filters.items() if v)
        ]
    else:
        valid_indices = list(range(len(index.chunks)))

    if not valid_indices:
        return []

    # Нормалізація вектора запиту
    q = np.asarray(query_vector, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm > 0:
        q = q / q_norm

    sub_vectors = index.vectors[valid_indices]
    scores = sub_vectors @ q

    candidates: list[tuple[int, float]] = []
    for idx, score in zip(valid_indices, scores):
        score_val = float(score)
        if threshold is not None and score_val < threshold:
            continue
        candidates.append((idx, score_val))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:top_k]

    return [Hit(chunk=index.chunks[idx], score=score) for idx, score in top_candidates]
