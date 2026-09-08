"""Модуль inference: єдине місце застосунку, яке знає про модель.

Тут живуть ваги, поріг упевненості й формат «сирого» результату моделі.
Веб-рівень (`app/main.py`) отримує звідси готовий структурований список
знайдених обʼєктів і нічого не знає ані про `ultralytics`, ані про те,
у якому вигляді модель віддає рамки.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі
рішення з розділу 2 практичної роботи:

* де саме завантажувати ваги, щоб це сталося **один раз**, а не на кожен запит;
* яким узяти поріг упевненості й чи дозволяти змінювати його ззовні;
* у якому вигляді віддавати результат: які поля, які одиниці координат;
* як виміряти час inference і що саме до нього зараховувати;
* як повестися, коли надійшов не той файл — не зображення або порожній.

Довідка про модель: https://docs.ultralytics.com/
"""

import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image, UnidentifiedImageError

from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "yolov8n.pt"
WEIGHTS = str(WEIGHTS_PATH) if WEIGHTS_PATH.exists() else "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25

_model: Optional[YOLO] = None


class DetectionError(Exception):
    """Помилка детекції, зрозуміла веб-рівню.

    Заготовка. Вирішіть, чи достатньо одного типу помилки, чи їх варто
    розрізняти — некоректний файл, збій моделі, — і що з цього має
    побачити користувач.
    """


class InvalidImageError(DetectionError):
    pass


class ModelInferenceError(DetectionError):
    pass


def load_model() -> YOLO:
    """Повернути готову до роботи модель.

    Завантаження ваг коштує дорого. Подумайте, як зробити так, щоб воно
    відбулося один раз за час життя застосунку.
    """
    global _model
    if _model is None:
        try:
            _model = YOLO(WEIGHTS)
        except Exception as exc:
            raise ModelInferenceError(f"Не вдалося завантажити ваги моделі: {exc}") from exc
    return _model


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, Any]:
    """Знайти обʼєкти на зображенні.

    Приймає байти завантаженого файлу, повертає структурований результат:
    для кожного знайденого обʼєкта — клас, рамку й упевненість, а також
    їхню кількість і час виконання. Точний склад полів — ваше рішення.
    """
    if not image_bytes or len(image_bytes.strip()) == 0:
        raise InvalidImageError("Надіслано порожній файл або файл нульового розміру.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        img_rgb = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(f"Файл не є дійсним або підтримуваним зображенням: {exc}") from exc

    if not (0.0 <= confidence <= 1.0):
        confidence = DEFAULT_CONFIDENCE

    model = load_model()

    start_time = time.perf_counter()
    try:
        results = model.predict(source=img_rgb, conf=confidence, verbose=False)
    except Exception as exc:
        raise ModelInferenceError(f"Збій під час інференсу моделі: {exc}") from exc
    duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

    objects: List[Dict[str, Any]] = []
    if results and len(results) > 0:
        result = results[0]
        for box in result.boxes:
            coords = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            conf_val = float(box.conf[0].item())
            class_name = model.names.get(cls_id, f"class_{cls_id}")

            objects.append({
                "class_name": class_name,
                "confidence": round(conf_val, 4),
                "box": {
                    "x1": round(coords[0], 1),
                    "y1": round(coords[1], 1),
                    "x2": round(coords[2], 1),
                    "y2": round(coords[3], 1),
                }
            })

    return {
        "objects": objects,
        "count": len(objects),
        "inference_time_ms": duration_ms,
        "confidence_threshold": confidence,
        "image_size": {
            "width": image.width,
            "height": image.height
        }
    }
