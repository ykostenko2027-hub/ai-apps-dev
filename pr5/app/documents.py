"""Колекція документів: читання, метадані, поділ на фрагменти.

Це єдине місце, яке знає, як влаштовані файли в `docs/`: де в них
метадані, як розмічено текст, за якими межами його ділити. Решта
застосунку працює з готовими фрагментами (`Chunk`) і не читає файлів.

Розбір блоку метаданих реалізовано: це формат файлів, а не предмет
роботи. Поділ на фрагменти — заготовка: розмір, межі, перекриття і те,
що саме потрапляє в кожен фрагмент, — рішення з розділу 2 практичної
роботи.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"

# Параметри поділу — відправна точка, а не рекомендація. У чому їх
# рахувати (символи, слова, токени моделі) і як застосовувати, коли
# ділите за заголовками, — ваше рішення.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — те, що перетворюється на вектор і показується в результатах.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — поля з блоку метаданих файлу (title, category, product,
    audience, updated, status) плюс те, що ви вирішите додати самі:
    заголовок розділу, порядковий номер фрагмента, позицію в документі.
    Фільтри пошуку працюють саме з цим словником.
    """

    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку
    файлу. Повертає словник метаданих і решту тексту. Порожні значення
    (`product:` без нічого) стають порожнім рядком. Якщо блоку немає —
    порожній словник і текст як є.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного
    `*.md` у папці, крім `README.md` — він описує колекцію, а не є її
    частиною. Порядок — за іменем файлу, щоб індекс будувався однаково
    від запуску до запуску.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    """Поділити текст документа на фрагменти.

    * Поділ за межами розділів (заголовки `## `) та абзацами `\n\n`.
    * Зберігає цілісність таблиць та списків помилок.
    * Додає назву документа та заголовок розділу до тексту та метаданих,
      забезпечуючи контекст навіть для коротких уривків.
    """
    import re

    doc_title = metadata.get("title", source)
    parts = re.split(r"(?m)^##\s+", text)
    chunks: list[Chunk] = []

    # Вступна частина до першого розділу (якщо є)
    intro = parts[0].strip()
    intro_lines = [line for line in intro.splitlines() if not line.startswith("# ")]
    intro_text = "\n".join(intro_lines).strip()
    if intro_text:
        meta = dict(metadata)
        meta["heading"] = "Вступ"
        chunk_text = f"[{doc_title} — Вступ]\n{intro_text}"
        chunks.append(Chunk(text=chunk_text, source=source, metadata=meta))

    # Обробка розділів другого рівня ##
    for part in parts[1:]:
        lines = part.splitlines()
        heading = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        if not content:
            continue

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        current_paras: list[str] = []
        current_len = 0

        for para in paragraphs:
            if current_len + len(para) > CHUNK_SIZE and current_paras:
                meta = dict(metadata)
                meta["heading"] = heading
                body = "\n\n".join(current_paras)
                chunk_text = f"[{doc_title} — {heading}]\n{body}"
                chunks.append(Chunk(text=chunk_text, source=source, metadata=meta))
                current_paras = [para]
                current_len = len(para)
            else:
                current_paras.append(para)
                current_len += len(para)

        if current_paras:
            meta = dict(metadata)
            meta["heading"] = heading
            body = "\n\n".join(current_paras)
            chunk_text = f"[{doc_title} — {heading}]\n{body}"
            chunks.append(Chunk(text=chunk_text, source=source, metadata=meta))

    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
