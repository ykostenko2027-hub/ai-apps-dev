"""Сховище стану звернень: зберегти, прочитати, перелічити.

Стан зберігається у JSON-файлах у каталозі RUNS_DIR.
Запис виконується атомарно (через тимчасовий файл і os.replace),
щоб уникнути пошкодження даних при аварійному завершенні процесу.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import uuid
from typing import Optional

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))


def _ensure_dir() -> None:
    """Переконатися, що каталог існує."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _deserialize_run(data: dict) -> Run:
    """Відновити об'єкт Run із словника."""
    steps_data = data.pop("steps", [])
    steps = [Step(**s) for s in steps_data]
    return Run(steps=steps, **data)


def save(run: Run) -> None:
    """Зберегти стан звернення, атомарно замінивши попередній файл."""
    _ensure_dir()
    target_path = RUNS_DIR / f"{run.run_id}.json"
    temp_path = RUNS_DIR / f".tmp-{run.run_id}-{uuid.uuid4().hex}.json"

    data = asdict(run)
    text = json.dumps(data, ensure_ascii=False, indent=2)

    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, target_path)


def load(run_id: str) -> Run | None:
    """Прочитати стан звернення; None, якщо такого немає."""
    target_path = RUNS_DIR / f"{run_id}.json"
    if not target_path.exists():
        return None
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _deserialize_run(data)
    except Exception:
        return None


def list_runs(status: str | None = None) -> list[Run]:
    """Усі звернення, за потреби — лише з певним статусом; нові першими."""
    _ensure_dir()
    runs: list[Run] = []
    
    files = [p for p in RUNS_DIR.glob("*.json") if not p.name.startswith(".")]
    # Сортуємо за часом модифікації файлу (найновіші спочатку)
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    for path in files:
        run_id = path.stem
        r = load(run_id)
        if r is not None:
            if status is None or r.status == status:
                runs.append(r)

    # Якщо доступне created_at, додатково відсортуємо за created_at спаданням
    runs.sort(key=lambda r: r.created_at or "", reverse=True)
    return runs
