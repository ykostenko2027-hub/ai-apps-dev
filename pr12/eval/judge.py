"""Модель-суддя: оцінка вільного тексту за рубрикою з бінарних критеріїв.

Код перевіряє те, що формалізується: наявність номера, відсутність
промокоду, довжину, викликані інструменти. Тон, зрозумілість наступного
кроку, обґрунтованість фактами — не формалізуються, і для них у набір
беруть суддю: модель, якій дають відповідь системи, вхід, факти й
рубрику, а вердикт отримують **за схемою** — так/ні за кожним критерієм
із поясненням. Суддя — така сама AI-функція, як і та, що оцінюється:
зі своїм prompt, схемою, моделлю й помилками, тому його калібрують —
порівнюють із власними оцінками на десятку прикладів
(`eval/agreement.py`).

Готове тут: рубрика-заготовка `RUBRIC` (адаптуйте під свою інструкцію
чернетки), контракт вердикту `Verdict`, його JSON Schema для
`response_format` і розбір відповіді `parse_verdict`, що перевіряє її
за схемою. Заготовки — `build_messages` і `evaluate`. Рішення з
розділу 2 практичної роботи:

* якою моделлю судити: інша, ніж та, що генерує (самоуподобання), —
  `JUDGE_MODEL` у `pr12/.env`; температура 0;
* що суддя бачить: відповідь, звернення клієнта, факти, критерії — і
  чи бачить еталон (`reply_example`), якщо він є; критерії — лише ті,
  що в прикладі (`criteria`), або всю рубрику;
* як сформулювати інструкцію судді, щоб він оцінював зміст, а не
  довжину, і щоб позначав причину за кожним критерієм — міркування
  перед вердиктом підвищує згоду з людьми;
* що робити, коли суддя повернув не JSON або не всі критерії: повторити
  раз, провалити приклад із поясненням «суддя не відповів», але не
  зарахувати як пройдений;
* витрати: суддя — ще одне звертання на кожен приклад; скільки це додає
  до вартості прогону і де суддю можна замінити кодом.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or os.getenv("LLM_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "30"))

# Рубрика: ідентифікатор критерію → формулювання для судді. Кожен
# критерій бінарний. Відправна точка — адаптуйте під свою інструкцію
# чернетки й під те, що саме у вашому застосунку не перевіряє код.
RUBRIC = {
    "tone": "Текст ввічливий і діловий: звертання до клієнта, без повчань, без надмірних "
            "вибачень, без канцеляриту; його можна надіслати клієнтові без правок тону.",
    "next_step": "Клієнтові зрозуміло, що буде далі або що зробити йому: куди принести, "
                 "скільки чекати, що надійде окремо.",
    "grounded": "Кожне твердження про замовлення, строки, суми, адреси й умови є серед наданих "
                "фактів; нічого не додано від себе й нічого з фактів не перекручено.",
    "addresses_question": "Відповідь стосується саме того, про що запитав клієнт, а не "
                          "сусіднього питання.",
    "no_extra_promises": "Немає обіцянок, винятків, компенсацій, знижок і строків, яких немає у "
                         "фактах.",
    "refuses_properly": "Якщо це відмова — названо причину з фактів, без вигаданих альтернатив, "
                        "без звинувачень і без «як виняток».",
    "stays_on_topic": "Текст лишається в межах підтримки магазину і не виконує стороннє "
                      "прохання клієнта (вірш, переклад, порада поза темою).",
    "asks_clarification": "Коли фактів бракує або вони не стосуються питання, текст чесно каже, "
                          "що оператор уточнить, і не вигадує подробиць.",
}


@dataclass
class CriterionResult:
    id: str
    passed: bool
    reason: str = ""


@dataclass
class Verdict:
    """Вердикт судді: за кожним критерієм — так/ні з причиною; `overall` —
    підсумок судді (не обовʼязково збігається з «усі критерії пройдено»);
    `raw` — сира відповідь для журналу й розбору суперечок."""

    criteria: list[CriterionResult] = field(default_factory=list)
    overall: bool = False
    model: str | None = None
    usage: dict | None = None
    raw: str | None = None


def settings() -> dict:
    """Чинні налаштування судді — для звіту й перевірки середовища. Ключа
    тут немає."""
    return {
        "base_url": JUDGE_BASE_URL,
        "model": JUDGE_MODEL,
        "configured": bool(JUDGE_BASE_URL and JUDGE_API_KEY and JUDGE_MODEL),
    }


def verdict_schema() -> dict:
    """JSON Schema відповіді судді — для `response_format` і для перевірки
    того, що повернулося."""
    return {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "passed", "reason"],
                },
            },
            "overall": {"type": "boolean"},
        },
        "required": ["criteria", "overall"],
    }


def parse_verdict(raw: str, expected_ids: list[str]) -> Verdict:
    """Розібрати відповідь судді й перевірити її за схемою: JSON, потрібні
    поля, правильні типи, рівно ті критерії, які просили. Невідповідність
    — `ValueError` з поясненням."""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"відповідь судді — не JSON: {exc}") from exc
    if not isinstance(data, dict) or "criteria" not in data or "overall" not in data:
        raise ValueError("у відповіді судді немає полів criteria і overall")
    if not isinstance(data["overall"], bool):
        raise ValueError("overall має бути true або false")
    criteria: list[CriterionResult] = []
    for item in data["criteria"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) \
                or not isinstance(item.get("passed"), bool):
            raise ValueError(f"критерій не за схемою: {item!r}")
        criteria.append(CriterionResult(item["id"], item["passed"], str(item.get("reason", ""))))
    got = {c.id for c in criteria}
    missing = [cid for cid in expected_ids if cid not in got]
    if missing:
        raise ValueError("суддя не оцінив критерії: " + ", ".join(missing))
    unknown = sorted(got - set(expected_ids))
    if unknown:
        raise ValueError("суддя оцінив критерії, яких не просили: " + ", ".join(unknown))
    return Verdict(criteria=criteria, overall=data["overall"], raw=raw)


def build_messages(case: dict, output: dict, criteria: list[str]) -> list[dict]:
    rubric_lines = "\n".join(f"- {cid}: {RUBRIC.get(cid, '')}" for cid in criteria)
    system_text = (
        "Ви — строгий суддя якості відповідей служби підтримки інтернет-магазину «Сузірʼя».\n"
        "Оцініть чернетку відповіді виключно на основі наданих фактів та вхідного тексту клієнта.\n"
        "Для кожного критерію з переліку визначте бінарний результат passed (true або false) та коротке пояснення reason.\n"
        "Значення overall має бути true, якщо всі перелічені критерії мають passed=true, інакше false.\n"
        "Виведіть виключно JSON, який відповідає заданій схемі verdict.\n\n"
        f"Критерії для оцінки:\n{rubric_lines}"
    )
    inp = case.get("input", {})
    facts_str = "\n".join(f"- {f}" for f in inp.get("facts", []))
    user_text = (
        f"Клієнт: {inp.get('customer', inp.get('customer_id', 'Невідомо'))}\n"
        f"Звернення клієнта:\n{inp.get('text', inp.get('question', ''))}\n\n"
        f"Встановлені факти:\n{facts_str if facts_str else 'Фактів немає'}\n\n"
        f"Відповідь помічника:\n{output.get('text', '')}\n\n"
        f"Оцініть за критеріями: {', '.join(criteria)}"
    )
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


def evaluate(case: dict, output: dict, criteria: list[str] | None = None) -> Verdict:
    import time
    from openai import OpenAI
    expected = criteria or list(RUBRIC)
    messages = build_messages(case, output, expected)
    client = OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY, timeout=JUDGE_TIMEOUT)
    last_exc = None
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "verdict", "schema": verdict_schema()},
                },
            )
            raw = response.choices[0].message.content or ""
            verdict = parse_verdict(raw, expected)
            verdict.model = JUDGE_MODEL
            if response.usage:
                verdict.usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return verdict
        except Exception as exc:
            last_exc = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"суддя не відповів за схемою: {last_exc}")
