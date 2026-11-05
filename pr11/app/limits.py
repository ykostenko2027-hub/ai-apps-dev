import os
import threading
import time
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))


class RateLimited(Exception):
    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = retry_after


_lock = threading.Lock()
_clients: dict[str, list[float]] = defaultdict(list)


def settings() -> dict:
    return {"per_minute": PER_MINUTE}


def acquire(client: str) -> None:
    now = time.time()
    window = 60.0
    with _lock:
        timestamps = [t for t in _clients[client] if now - t < window]
        if len(timestamps) >= PER_MINUTE:
            oldest = timestamps[0]
            retry_after = max(1.0, window - (now - oldest))
            raise RateLimited(retry_after)
        timestamps.append(now)
        _clients[client] = timestamps
