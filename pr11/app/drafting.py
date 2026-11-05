import os
from dotenv import load_dotenv

from .providers.base import Request

load_dotenv()

PROMPT_VERSION = "drafts-v1"


def build_request(sample: dict) -> Request:
    system_prompt = (
        "Ти — оператор підтримки інтернет-магазину «Сузірʼя». "
        "Відповідай ввічливо, українською мовою, спираючись виключно на наведені факти. "
        "Якщо фактів недостатньо, ввічливо повідоми про це."
    )
    facts_text = "\n".join(f"- {fact}" for fact in sample.get("facts", []))
    user_prompt = f"Клієнт: {sample.get('text', '')}\n\nФакти:\n{facts_text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    metadata = {
        "request_id": sample.get("id", ""),
        "prompt_version": PROMPT_VERSION,
        "facts_count": len(sample.get("facts", [])),
        "text_len": len(sample.get("text", "")),
    }

    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "600"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    return Request(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        metadata=metadata,
    )


def check(text: str, sample: dict) -> list[str]:
    warnings: list[str] = []
    if not text.strip():
        warnings.append("Порожня чернетка")
    return warnings
