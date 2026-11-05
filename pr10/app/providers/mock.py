import os
import random
import time
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result
from .. import samples

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")


class MockProvider:
    name = "mock"
    model = "mock-1"

    def _get_text(self, request: Request) -> str:
        req_id = request.metadata.get("request_id")
        if req_id:
            s = samples.get(req_id)
            if s and "reply_example" in s:
                return s["reply_example"]
        return "Добрий день! Дякуємо за звернення до служби підтримки «Сузірʼя». Ваша заявка опрацьовується."

    def _check_failure(self) -> None:
        if FAILURE_RATE > 0 and random.random() < FAILURE_RATE:
            if FAILURE_KIND == "timeout":
                time.sleep((FIRST_DELAY_MS / 1000.0) + 1.0)
                raise ProviderError("timeout", "Таймаут очікування відповіді від провайдера", retryable=True)
            elif FAILURE_KIND == "rate_limit":
                raise ProviderError("rate_limit", "Перевищено ліміт запитів провайдера", retryable=True, retry_after=2.0)
            elif FAILURE_KIND == "auth":
                raise ProviderError("auth", "Помилка автентифікації провайдера", retryable=False)
            else:
                raise ProviderError("unavailable", "Провайдер тимчасово недоступний", retryable=True)

    def complete(self, request: Request) -> Result:
        t0 = time.perf_counter()
        if FIRST_DELAY_MS > 0:
            time.sleep(FIRST_DELAY_MS / 1000.0)
        self._check_failure()
        text = self._get_text(request)
        tokens = text.split(" ")
        if DELAY_MS > 0:
            time.sleep((len(tokens) * DELAY_MS) / 1000.0)
        elapsed = time.perf_counter() - t0
        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage={"prompt_tokens": 100, "completion_tokens": len(tokens), "total_tokens": 100 + len(tokens)},
            finish_reason="stop",
            elapsed=round(elapsed, 2),
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        t0 = time.perf_counter()
        if FIRST_DELAY_MS > 0:
            time.sleep(FIRST_DELAY_MS / 1000.0)
        self._check_failure()
        text = self._get_text(request)
        words = text.split(" ")
        tokens = [w + " " for w in words[:-1]] + [words[-1]]
        should_fail_mid = (FAILURE_RATE > 0 and FAILURE_KIND == "mid_stream" and random.random() < FAILURE_RATE)

        for i, token in enumerate(tokens):
            if should_fail_mid and i >= 3:
                raise ProviderError("unavailable", "Обрив з'єднання під час потокового передавання", retryable=True)
            if DELAY_MS > 0 and i > 0:
                time.sleep(DELAY_MS / 1000.0)
            is_last = (i == len(tokens) - 1)
            yield Chunk(
                text=token,
                usage={"prompt_tokens": 100, "completion_tokens": len(tokens), "total_tokens": 100 + len(tokens)} if is_last else None,
                finish_reason="stop" if is_last else None,
            )
