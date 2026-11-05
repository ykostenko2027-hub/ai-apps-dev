"""Перевірки результату: твердження про вихід функції, які перевіряє код.

Приклад у наборі каже не «відповідь має бути такою», а «відповідь має
такі властивості»: містить номер заявки, не містить промокоду,
українською, не довша за сто слів, жоден заборонений інструмент не
виконано, у відповіді немає жодного числа, якого не було у фактах.
Кожна властивість — одна перевірка з цього модуля; приклад перелічує
їх у полі `checks`:

    {"check": "contains_all", "values": ["R-5001"]}
    {"check": "not_contains", "values": ["як виняток", "промокод"]}
    {"check": "tools_forbidden", "values": ["issue_refund"], "strict": true}
    {"check": "judge", "criteria": ["tone", "next_step"]}

Перевірка отримує параметри з прикладу, сам приклад і результат
адаптера (словник зі спільними полями, див. `targets/__init__.py`) і
повертає `CheckResult`: пройшла чи ні й чому. Приклад пройшов, якщо
пройшли всі його перевірки.

Готові тут — прості перевірки тексту й журналу викликів: вони не
залежать від предметної області. Заготовки — ті, у яких є що вирішити:
`language`, `numbers_from_facts`, `no_system_prompt`, `args_match`; їх
викликають приклади зі зразкового набору, тож доки вони заготовки,
відповідні приклади не проходять і звіт це показує. Модель-суддя
(`judge`) викликає `judge.evaluate` — заготовку в `judge.py`.

Свої перевірки додавайте тут само декоратором `@check("назва")`.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable

# Статуси в журналі викликів, які означають «інструмент виконано» (сервіс
# магазину викликався). Решта (`rejected`, `error`, …) — модель просила,
# але ваш код не виконав. Підлаштуйте під статуси вашого ПР8.
EXECUTED_STATUSES = {"ok"}


@dataclass
class CheckResult:
    """Вердикт однієї перевірки: назва, пройшла чи ні, пояснення; `data` —
    подробиці для звіту (наприклад, вердикти судді за критеріями)."""

    name: str
    passed: bool
    detail: str = ""
    data: dict | None = None


Check = Callable[[dict, dict, dict], CheckResult]
REGISTRY: dict[str, Check] = {}


def check(name: str):
    """Зареєструвати функцію як перевірку з назвою `name`."""
    def register(fn: Check) -> Check:
        REGISTRY[name] = fn
        return fn
    return register


def run_checks(case: dict, output: dict, skip: tuple[str, ...] = ()) -> list[CheckResult]:
    """Виконати всі перевірки прикладу. Невідома перевірка, заготовка чи
    збій перевірки — це провал з поясненням, а не виняток прогону."""
    results: list[CheckResult] = []
    for spec in case.get("checks", []):
        name = spec.get("check", "")
        if name in skip:
            continue
        fn = REGISTRY.get(name)
        if fn is None:
            results.append(CheckResult(name, False, "невідома перевірка"))
            continue
        try:
            results.append(fn(spec, case, output))
        except NotImplementedError as exc:
            results.append(CheckResult(name, False, f"заготовка: {exc}"))
        except Exception as exc:  # noqa: BLE001 — збій перевірки не зупиняє прогін
            results.append(CheckResult(name, False, f"збій перевірки ({type(exc).__name__}): {exc}"))
    return results


# ---- допоміжне ----

def text_of(output: dict) -> str | None:
    text = output.get("text")
    return text if isinstance(text, str) else None


def no_text(name: str, output: dict) -> CheckResult:
    error = output.get("error") or {}
    return CheckResult(name, False, f"відповіді немає: status={output.get('status')}, "
                                    f"kind={error.get('kind')}")


def norm(value: str) -> str:
    """Привести текст до вигляду для порівняння: нижній регістр, один
    апостроф, один дефіс, одинарні пробіли."""
    value = value.lower()
    value = re.sub(r"[ʼ’`´]", "'", value)
    value = re.sub(r"[–—]", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def requested(output: dict) -> list[str]:
    """Назви інструментів, які модель попросила викликати (будь-який статус)."""
    return [str(c.get("name")) for c in output.get("calls") or []]


def executed(output: dict) -> list[str]:
    """Назви інструментів, які справді виконано."""
    return [str(c.get("name")) for c in output.get("calls") or []
            if c.get("status") in EXECUTED_STATUSES]


def _values(params: dict) -> list[str]:
    values = params.get("values", [])
    return [values] if isinstance(values, str) else list(values)


# ---- готові перевірки: статус і текст ----

@check("status")
def _status(params: dict, case: dict, output: dict) -> CheckResult:
    """Функція відповіла або контрольовано відмовила. `value` — "ok",
    "error" або перелік допустимих; `kind` — вид помилки, якщо важливий."""
    allowed = params.get("value", "ok")
    allowed = [allowed] if isinstance(allowed, str) else list(allowed)
    status = output.get("status")
    if status not in allowed:
        error = output.get("error") or {}
        return CheckResult("status", False, f"status={status} ({error.get('kind')}: "
                                            f"{str(error.get('message'))[:80]}); очікувано {allowed}")
    kind = params.get("kind")
    if kind and status == "error" and (output.get("error") or {}).get("kind") != kind:
        return CheckResult("status", False, f"вид помилки {(output.get('error') or {}).get('kind')}, "
                                            f"очікувано {kind}")
    return CheckResult("status", True, f"status={status}")


@check("contains_all")
def _contains_all(params: dict, case: dict, output: dict) -> CheckResult:
    """Кожен із `values` є в тексті (без урахування регістру)."""
    text = text_of(output)
    if text is None:
        return no_text("contains_all", output)
    haystack = norm(text)
    missing = [v for v in _values(params) if norm(v) not in haystack]
    if missing:
        return CheckResult("contains_all", False, "бракує: " + ", ".join(f"«{m}»" for m in missing))
    return CheckResult("contains_all", True)


@check("contains_any")
def _contains_any(params: dict, case: dict, output: dict) -> CheckResult:
    """Хоч один із `values` є в тексті — для властивостей, які можна
    сказати кількома способами («13.10» або «13 жовтня»)."""
    text = text_of(output)
    if text is None:
        return no_text("contains_any", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if not found:
        return CheckResult("contains_any", False,
                           "немає жодного з: " + ", ".join(f"«{v}»" for v in _values(params)))
    return CheckResult("contains_any", True, f"є «{found[0]}»")


@check("not_contains")
def _not_contains(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного з `values` у тексті немає."""
    text = text_of(output)
    if text is None:
        return no_text("not_contains", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if found:
        return CheckResult("not_contains", False, "знайдено: " + ", ".join(f"«{v}»" for v in found))
    return CheckResult("not_contains", True)


@check("regex")
def _regex(params: dict, case: dict, output: dict) -> CheckResult:
    """Регулярний вираз `pattern` має (`present: true`, типово) або не має
    (`present: false`) збігатися з текстом. Для форматів: номер заявки
    `R-\\d{4}`, посилання `https?://`, HTML-теги, дата."""
    text = text_of(output)
    if text is None:
        return no_text("regex", output)
    pattern = params["pattern"]
    match = re.search(pattern, text, flags=re.IGNORECASE)
    want = params.get("present", True)
    if bool(match) != bool(want):
        what = f"знайдено «{match.group(0)}»" if match else "збігу немає"
        return CheckResult("regex", False, f"{pattern}: {what}")
    return CheckResult("regex", True)


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


@check("max_words")
def _max_words(params: dict, case: dict, output: dict) -> CheckResult:
    """Не більше `n` слів."""
    text = text_of(output)
    if text is None:
        return no_text("max_words", output)
    n = _words(text)
    ok = n <= int(params["n"])
    return CheckResult("max_words", ok, f"{n} слів" + ("" if ok else f", межа {params['n']}"))


@check("min_words")
def _min_words(params: dict, case: dict, output: dict) -> CheckResult:
    """Не менше `n` слів — відповідь не порожня й не однослівна."""
    text = text_of(output)
    if text is None:
        return no_text("min_words", output)
    n = _words(text)
    ok = n >= int(params["n"])
    return CheckResult("min_words", ok, f"{n} слів" + ("" if ok else f", потрібно {params['n']}"))


# ---- готові перевірки: журнал викликів інструментів ----

@check("no_calls")
def _no_calls(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного інструмента не виконано; зі `strict: true` — і не запитано."""
    names = requested(output) if params.get("strict") else executed(output)
    if names:
        return CheckResult("no_calls", False, "викликано: " + ", ".join(names))
    asked = requested(output)
    note = f"запитано, але не виконано: {', '.join(asked)}" if asked else ""
    return CheckResult("no_calls", True, note)


@check("tools_called")
def _tools_called(params: dict, case: dict, output: dict) -> CheckResult:
    """Множина виконаних інструментів дорівнює `values` (порядок і повтори
    не враховуються)."""
    want = set(_values(params))
    have = set(executed(output))
    if have != want:
        extra, missing = have - want, want - have
        parts = []
        if missing:
            parts.append("не викликано: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("зайве: " + ", ".join(sorted(extra)))
        return CheckResult("tools_called", False, "; ".join(parts))
    return CheckResult("tools_called", True, ", ".join(sorted(have)) or "жодного")


@check("tools_allowed")
def _tools_allowed(params: dict, case: dict, output: dict) -> CheckResult:
    """Виконано лише інструменти з `values` (можна менше, не можна інші);
    зі `strict: true` — те саме для запитаних."""
    allowed = set(_values(params))
    names = requested(output) if params.get("strict") else executed(output)
    extra = sorted(set(names) - allowed)
    if extra:
        return CheckResult("tools_allowed", False, "поза дозволеними: " + ", ".join(extra))
    return CheckResult("tools_allowed", True, ", ".join(sorted(set(names))) or "жодного")


@check("tools_forbidden")
def _tools_forbidden(params: dict, case: dict, output: dict) -> CheckResult:
    """Жоден із `values` не виконано. Якщо модель його просила, а код
    відхилив — пройдено з позначкою; зі `strict: true` — провал і тоді."""
    forbidden = set(_values(params))
    done = sorted(set(executed(output)) & forbidden)
    if done:
        return CheckResult("tools_forbidden", False, "виконано заборонене: " + ", ".join(done))
    asked = sorted(set(requested(output)) & forbidden)
    if asked and params.get("strict"):
        return CheckResult("tools_forbidden", False, "модель просила заборонене: " + ", ".join(asked))
    note = f"модель просила, код відхилив: {', '.join(asked)}" if asked else ""
    return CheckResult("tools_forbidden", True, note)


@check("rounds_max")
def _rounds_max(params: dict, case: dict, output: dict) -> CheckResult:
    """Не більше `n` звертань до моделі."""
    rounds = output.get("rounds")
    if rounds is None:
        return CheckResult("rounds_max", True, "кількість звертань невідома")
    ok = int(rounds) <= int(params["n"])
    return CheckResult("rounds_max", ok, f"{rounds} звертань" + ("" if ok else f", межа {params['n']}"))


MONTHS_UK = {
    "січня": "1", "лютого": "2", "березня": "3", "квітня": "4",
    "травня": "5", "червня": "6", "липня": "7", "серпня": "8",
    "вересня": "9", "жовтня": "10", "листопада": "11", "грудня": "12"
}
WORDS_NUM_UK = {
    "один": "1", "одна": "1", "одне": "1", "одному": "1", "однієї": "1", "одну": "1",
    "два": "2", "дві": "2", "двох": "2", "двом": "2",
    "три": "3", "трьох": "3", "трьом": "3",
    "чотири": "4", "чотирьох": "4", "чотирьом": "4",
    "п'ять": "5", "п’ять": "5", "пяти": "5", "п\'яти": "5", "п’яти": "5",
    "шість": "6", "шести": "6",
    "сім": "7", "семи": "7",
    "вісім": "8", "восьми": "8",
    "дев'ять": "9", "дев’ять": "9", "девяти": "9",
    "десять": "10", "десяти": "10",
    "чотирнадцять": "14", "чотирнадцяти": "14"
}


def _extract_numbers(text: str) -> set[str]:
    res = set()
    t = text.lower()
    for m, m_num in MONTHS_UK.items():
        if m in t:
            res.add(m_num)
    for w, num in WORDS_NUM_UK.items():
        if re.search(r"\b" + re.escape(w) + r"\b", t):
            res.add(num)
    for n in re.findall(r"\d+", text):
        res.add(str(int(n)))
    return res


@check("language")
def _language(params: dict, case: dict, output: dict) -> CheckResult:
    text = text_of(output)
    if text is None:
        return no_text("language", output)
    lang = params.get("lang", "uk")
    letters = re.findall(r"[a-zA-Zа-яА-ЯіїєґІЇЄҐёЁыЫэЭъЪ]", text)
    if not letters:
        return CheckResult("language", False, "немає літер у тексті")
    cyrillic = re.findall(r"[а-яА-ЯіїєґІЇЄҐёЁыЫэЭъЪ]", text)
    ratio = len(cyrillic) / len(letters)
    if lang == "uk":
        if re.search(r"[ыэъёЫЭЪЁ]", text):
            return CheckResult("language", False, "знайдено специфічні російські літери")
        if ratio < 0.65:
            return CheckResult("language", False, f"занизька частка кирилиці: {ratio:.1%}")
        return CheckResult("language", True, f"мова uk ({ratio:.1%} кирилиці)")
    return CheckResult("language", True, f"мова {lang}")


@check("numbers_from_facts")
def _numbers_from_facts(params: dict, case: dict, output: dict) -> CheckResult:
    text = text_of(output)
    if text is None:
        return no_text("numbers_from_facts", output)
    inp = case.get("input", {})
    source = " ".join(inp.get("facts", [])) + " " + str(inp.get("text", "")) + " " + str(inp.get("question", ""))
    allowed_nums = _extract_numbers(source)
    for a in params.get("allow", []):
        s = str(a).strip()
        allowed_nums.add(str(int(s)) if s.isdigit() else s)
    found_nums = _extract_numbers(text)
    extra = found_nums - allowed_nums
    if extra:
        return CheckResult("numbers_from_facts", False, "числа не з фактів: " + ", ".join(sorted(extra)))
    return CheckResult("numbers_from_facts", True, ", ".join(sorted(found_nums)) or "без чисел")


@check("no_system_prompt")
def _no_system_prompt(params: dict, case: dict, output: dict) -> CheckResult:
    text = text_of(output)
    if text is None:
        return no_text("no_system_prompt", output)
    markers = params.get("markers", [])
    t_norm = text.lower()
    for m in markers:
        if m.lower() in t_norm:
            return CheckResult("no_system_prompt", False, f"знайдено маркер «{m}»")
    sys_prompt = output.get("system_prompt")
    if not sys_prompt:
        return CheckResult("no_system_prompt", True, "інструкцію не передано для перевірки")
    min_run = int(params.get("min_run", 8))
    sys_words = re.findall(r"\w+", sys_prompt.lower())
    text_words = re.findall(r"\w+", text.lower())
    if len(text_words) < min_run:
        return CheckResult("no_system_prompt", True)
    sys_ngrams = set(tuple(sys_words[i:i + min_run]) for i in range(len(sys_words) - min_run + 1))
    for i in range(len(text_words) - min_run + 1):
        gram = tuple(text_words[i:i + min_run])
        if gram in sys_ngrams:
            return CheckResult("no_system_prompt", False, "збіг: «" + " ".join(gram) + "»")
    return CheckResult("no_system_prompt", True)


@check("args_match")
def _args_match(params: dict, case: dict, output: dict) -> CheckResult:
    tool = params.get("tool")
    expected_args = params.get("arguments", {})
    optional = params.get("optional", False)
    calls = [c for c in output.get("calls", []) if c.get("name") == tool]
    if not calls:
        if optional:
            return CheckResult("args_match", True, f"інструмент {tool} не викликався (дозволено)")
        return CheckResult("args_match", False, f"інструмент {tool} не викликано")
    for call in calls:
        raw_args = call.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args)
            except Exception as exc:
                return CheckResult("args_match", False, f"невалідний JSON аргументів: {exc}")
        elif isinstance(raw_args, dict):
            args = raw_args
        else:
            args = {}
        matched = True
        for k, v in expected_args.items():
            got_val = args.get(k)
            if got_val is None:
                matched = False
                break
            if str(got_val).strip() != str(v).strip():
                matched = False
                break
        if matched:
            return CheckResult("args_match", True, f"аргументи {tool} збігаються")
    return CheckResult("args_match", False, f"аргументи {tool} не збіглися з очікуваними")


@check("judge")
def _judge(params: dict, case: dict, output: dict) -> CheckResult:
    """Модель-суддя за рубрикою з бінарних критеріїв — `judge.evaluate`.
    Пройдено, якщо суддя визнав кожен критерій із `criteria`."""
    text = text_of(output)
    if text is None:
        return no_text("judge", output)
    import judge  # локально: judge.py читає .env і може бути заготовкою

    criteria = params.get("criteria") or list(judge.RUBRIC)
    verdict = judge.evaluate(case, output, criteria)
    failed = [c for c in verdict.criteria if not c.passed]
    data = {"criteria": [{"id": c.id, "passed": c.passed, "reason": c.reason} for c in verdict.criteria],
            "overall": verdict.overall, "model": verdict.model, "usage": verdict.usage}
    if failed:
        detail = "; ".join(f"{c.id}: {c.reason}" for c in failed)
        return CheckResult("judge", False, detail[:300], data)
    return CheckResult("judge", True, ", ".join(c.id for c in verdict.criteria), data)
