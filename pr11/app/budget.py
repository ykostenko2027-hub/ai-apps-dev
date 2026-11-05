import datetime
import json
import logging
import os
import threading
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")

log = logging.getLogger("app.budget")
_lock = threading.Lock()
_current_day: str = ""
_spent: float = 0.0


class BudgetExceeded(Exception):
    def __init__(self, spent: float, limit: float, resets_at: str) -> None:
        super().__init__(
            f"денний ліміт витрат {limit:.2f} $ вичерпано (витрачено {spent:.4f} $); відновиться {resets_at}"
        )
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at


def _today_str() -> str:
    return datetime.date.today().isoformat()


def _resets_at_str() -> str:
    now = datetime.datetime.now().astimezone()
    tomorrow = now.date() + datetime.timedelta(days=1)
    reset_dt = datetime.datetime.combine(tomorrow, datetime.time.min, tzinfo=now.tzinfo)
    return reset_dt.isoformat()


def _ensure_loaded() -> None:
    global _current_day, _spent
    today = _today_str()
    if _current_day == today:
        return

    _current_day = today
    _spent = 0.0
    if not BUDGET_FILE:
        return

    p = Path(BUDGET_FILE)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("day") == today:
                _spent = float(data.get("spent", 0.0))
        except Exception:
            _spent = 0.0


def _save_disk() -> None:
    if not BUDGET_FILE:
        return
    p = Path(BUDGET_FILE)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp_path = p.with_suffix(".tmp")
    data = {"day": _current_day, "spent": round(_spent, 6)}
    try:
        temp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(p)
    except Exception:
        pass


def settings() -> dict:
    return {"daily_usd": DAILY_USD, "warn_at": WARN_AT, "file": BUDGET_FILE}


def check(expected_cost: float | None = None) -> None:
    with _lock:
        _ensure_loaded()
        resets_at = _resets_at_str()
        if _spent >= DAILY_USD:
            raise BudgetExceeded(_spent, DAILY_USD, resets_at)
        if expected_cost and (_spent + expected_cost > DAILY_USD):
            raise BudgetExceeded(_spent, DAILY_USD, resets_at)
        if _spent >= (DAILY_USD * WARN_AT):
            log.warning("budget.warning", extra={
                "spent": round(_spent, 4),
                "limit": DAILY_USD,
                "warn_at": WARN_AT,
            })


def record(cost: float | None) -> None:
    global _spent
    if cost is None or cost <= 0:
        return
    with _lock:
        _ensure_loaded()
        was_below_warn = _spent < (DAILY_USD * WARN_AT)
        _spent += float(cost)
        _save_disk()
        if was_below_warn and _spent >= (DAILY_USD * WARN_AT):
            log.warning("budget.warning", extra={
                "spent": round(_spent, 4),
                "limit": DAILY_USD,
                "warn_at": WARN_AT,
            })


def status() -> dict:
    with _lock:
        _ensure_loaded()
        resets_at = _resets_at_str()
        warning = _spent >= (DAILY_USD * WARN_AT)
        exceeded = _spent >= DAILY_USD
        return {
            "day": _current_day,
            "spent": round(_spent, 4),
            "limit": DAILY_USD,
            "remaining": max(0.0, round(DAILY_USD - _spent, 4)),
            "warning": warning,
            "exceeded": exceeded,
            "resets_at": resets_at,
        }
