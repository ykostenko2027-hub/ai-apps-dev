"""Пошук за ключовими словами — точка порівняння для семантичного.

Той самий набір фрагментів, той самий формат влучень (`Hit`), ті самі
фільтри — інший спосіб ранжувати. Без цього модуля не буде з чим
порівнювати семантичний пошук, а без порівняння — не буде відповіді на
питання, чи він узагалі потрібен для цієї колекції.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення:

* як розбивати текст на слова: що робити з регістром, розділовими
  знаками, апострофом у «звʼязок», числами й артикулами на кшталт
  `OR-X2-BLK`;
* чи зводити слова до основи — і чим, якщо так; без цього «повернути»
  й «повернення» для пошуку різні слова;
* чим ранжувати: кількістю збігів, TF-IDF, BM25 (пакет `rank_bm25`
  вже в залежностях);
* у яких одиницях оцінка й чи можна її порівнювати з оцінкою
  семантичного пошуку.
"""

from dataclasses import dataclass, field

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами."""

    chunks: list[Chunk]
    extra: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Враховує українські апострофи та дефіси в артикулах/моделях.
    """
    import re

    normalized = text.lower().replace("'", "ʼ").replace("’", "ʼ").replace("`", "ʼ")
    return re.findall(r"[a-zа-яіїєґ0-9]+(?:[-ʼ][a-zа-яіїєґ0-9]+)*", normalized)


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    from rank_bm25 import BM25Okapi

    tokenized_corpus = [tokenize(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    return KeywordIndex(chunks=chunks, extra={"bm25": bm25})


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту за допомогою BM25.

    Застосовує ті самі фільтри за метаданими до ранжування.
    """
    if not index or not index.chunks or "bm25" not in index.extra:
        return []

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

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    bm25 = index.extra["bm25"]
    doc_scores = bm25.get_scores(query_tokens)

    candidates: list[tuple[int, float]] = []
    for idx in valid_indices:
        score = float(doc_scores[idx])
        if score > 0:
            candidates.append((idx, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    top_candidates = candidates[:top_k]

    return [Hit(chunk=index.chunks[idx], score=score) for idx, score in top_candidates]
