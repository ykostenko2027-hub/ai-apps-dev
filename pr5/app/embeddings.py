"""Модуль ембедінгів: єдине місце застосунку, яке знає, яка модель
перетворює текст на вектор і як її викликати.

Індекс (`app/index.py`) і веб-рівень (`app/main.py`) отримують звідси
готові вектори й не знають, локальна це модель чи API. Замінивши модель
тут, ви не змінюєте решту коду — але маєте перебудувати індекс: вектори
різних моделей непорівнянні.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яку модель узяти й чому саме її: мови, розмірність, довжина входу,
  розмір, швидкість на CPU;
* чи потрібні моделі префікси для запиту й для тексту (у деяких моделей
  запит і документ кодуються по-різному — дивіться картку моделі);
* чи нормалізувати вектори і де — тут чи в індексі;
* як кодувати кілька сотень фрагментів: по одному чи пакетами.

Назва моделі читається з `.env`; значення за замовчуванням — відправна
точка, а не рекомендація.
"""

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Назва моделі ембедінгів. Для локальної моделі — ідентифікатор на
# Hugging Face Hub; для моделі через API — назва в провайдера.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

_model = None


def get_model():
    """Повернути готову до роботи модель (створюється один раз)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _get_prefix(kind: str) -> str:
    """Отримати префікс для моделей типу E5."""
    if "e5" in MODEL_NAME.lower():
        return f"{kind}: "
    return ""


def embed_passages(texts: list[str]) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Використовує префікс 'passage: ' (якщо вимагає модель) та повертає
    L2-нормалізований масив розмірності (N × dim).
    """
    model = get_model()
    prefix = _get_prefix("passage")
    prefixed_texts = [prefix + t for t in texts]
    vectors = model.encode(
        prefixed_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на нормалізований вектор."""
    model = get_model()
    prefix = _get_prefix("query")
    vector = model.encode(
        prefix + text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vector, dtype=np.float32)
