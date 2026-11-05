from typing import Iterator
import openai

from .base import Chunk, ProviderError, Request, Result


class OpenAICompatProvider:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout, max_retries=0)

    def _convert_error(self, exc: Exception) -> ProviderError:
        if isinstance(exc, openai.APITimeoutError):
            return ProviderError(kind="timeout", message="Час очікування відповіді провайдера вичерпано", retryable=True)
        if isinstance(exc, openai.RateLimitError):
            return ProviderError(kind="rate_limit", message="Перевищено ліміт запитів до моделі", retryable=True, retry_after=5.0)
        if isinstance(exc, openai.AuthenticationError):
            return ProviderError(kind="auth", message="Помилка автентифікації ключа доступу", retryable=False)
        if isinstance(exc, openai.BadRequestError):
            return ProviderError(kind="bad_request", message="Некоректний запит до моделі", retryable=False)
        if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError)):
            return ProviderError(kind="unavailable", message="Сервіс моделі тимчасово недоступний", retryable=True)
        return ProviderError(kind="unknown", message=str(exc), retryable=False)

    def complete(self, request: Request) -> Result:
        try:
            resp = self.client.chat.completions.create(
                model=request.model or self.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                stream=False,
            )
            choice = resp.choices[0]
            text = choice.message.content or ""
            usage = resp.usage.model_dump(exclude_none=True) if getattr(resp, "usage", None) else None
            return Result(
                text=text,
                provider=self.name,
                model=resp.model or self.model,
                usage=usage,
                finish_reason=choice.finish_reason or "stop",
                cached=False,
            )
        except Exception as exc:
            raise self._convert_error(exc) from exc

    def stream(self, request: Request) -> Iterator[Chunk]:
        model = request.model or self.model
        kwargs = {
            "model": model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            try:
                stream_iter = self.client.chat.completions.create(**kwargs)
            except openai.BadRequestError:
                kwargs.pop("stream_options", None)
                stream_iter = self.client.chat.completions.create(**kwargs)

            for part in stream_iter:
                text = ""
                finish_reason = None
                if part.choices and len(part.choices) > 0:
                    delta = part.choices[0].delta
                    text = delta.content or ""
                    finish_reason = part.choices[0].finish_reason

                usage = None
                if getattr(part, "usage", None):
                    usage = part.usage.model_dump(exclude_none=True) if hasattr(part.usage, "model_dump") else dict(part.usage)

                yield Chunk(text=text, usage=usage, finish_reason=finish_reason)
        except Exception as exc:
            raise self._convert_error(exc) from exc
