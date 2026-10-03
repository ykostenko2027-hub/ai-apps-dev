import math
from typing import List, Tuple, Dict, Any


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 3.0)


def filter_and_format_chunks(
        raw_results: List[Dict[str, Any]],
        top_k_for_model: int = 4,
        similarity_threshold: float = 0.35,
        max_token_budget: int = 1500
) -> Tuple[List[Dict[str, Any]], str]:
    filtered: List[Dict[str, Any]] = []

    for item in raw_results:
        meta = item.get("metadata", {})

        audience = meta.get("audience", "").lower()
        if audience in ["staff", "operator", "internal", "персонал"]:
            continue

        status = meta.get("status", "").lower()
        if status in ["archive", "archived", "архів"]:
            continue

        score = item.get("score", 0.0)
        if score < similarity_threshold:
            continue

        filtered.append(item)

    selected = filtered[:top_k_for_model]

    context_blocks = []
    final_chunks = []
    current_tokens = 0

    for idx, chunk in enumerate(selected, start=1):
        meta = chunk.get("metadata", {})
        title = meta.get("title", "Документ")
        section = meta.get("section", "Розділ")
        date_str = meta.get("date", "не вказано")
        text = chunk.get("text", "").strip()

        header = f"[{idx}] {title} › {section} (Редакція: {date_str}):"
        block = f"{header}\n«{text}»\n"

        block_tokens = estimate_tokens(block)
        if current_tokens + block_tokens > max_token_budget and final_chunks:
            break

        current_tokens += block_tokens
        context_blocks.append(block)

        chunk_info = dict(chunk)
        chunk_info["context_id"] = idx
        final_chunks.append(chunk_info)

    assembled_context = "\n".join(context_blocks)
    return final_chunks, assembled_context