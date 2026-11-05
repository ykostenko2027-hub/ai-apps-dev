from .providers.base import Request
from . import gateway


def build_request(sample: dict) -> Request:
    system_prompt = (
        "Ти — помічник оператора підтримки інтернет-магазину «Сузірʼя». "
        "Склади ввічливу, точну та лаконічну чернетку відповіді клієнтові українською мовою. "
        "Спирайся виключно на наведені факти. "
        "Не вигадуй жодних додаткових дат, умов чи альтернатив, яких немає у фактах."
    )
    facts_block = "\n".join(f"- {fact}" for fact in sample.get("facts", []))
    user_prompt = (
        f"Звернення клієнта ({sample.get('customer', 'Клієнт')}):\n"
        f"«{sample.get('text', '')}»\n\n"
        f"Встановлені факти:\n{facts_block}\n\n"
        "Склади текст чернетки відповіді:"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    return Request(
        messages=messages,
        max_tokens=gateway.MAX_TOKENS,
        temperature=gateway.TEMPERATURE,
        metadata={"request_id": sample.get("id", "")},
    )


def check(text: str, sample: dict) -> list[str]:
    warnings = []
    if len(text.strip()) < 30:
        warnings.append("Чернетка занадто коротка.")
    for fact in sample.get("facts", []):
        for word in fact.split():
            clean_word = word.strip(".,;:!?\"'«»()")
            if clean_word.startswith("R-") or (clean_word.isdigit() and len(clean_word) >= 5):
                if clean_word not in text:
                    warnings.append(f"У чернетці не згадано номер/ідентифікатор: {clean_word}")
    return warnings
