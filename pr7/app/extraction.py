from dataclasses import dataclass, field
import time

from .images import prepare, ImageError
from .llm import extract, LLMError
from .rules import check, Issue


@dataclass
class Result:
    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def decide(document: dict | None, issues: list[Issue]) -> tuple[str, list[str]]:
    if document is None:
        return "reject", ["Не вдалося отримати дані з документа."]
    for issue in issues:
        if issue.rule == "document_type":
            return "reject", [issue.message]
    if issues:
        reasons = [issue.message for issue in issues]
        return "review", reasons
    return "auto", ["Усі обовʼязкові поля заповнено коректно, дані перевірено за довідниками та правилами."]


def process(content: bytes) -> Result:
    t0 = time.perf_counter()
    prepared = prepare(content)
    t_prep = time.perf_counter() - t0

    t1 = time.perf_counter()
    llm_result = extract(prepared)
    t_llm = time.perf_counter() - t1

    doc = llm_result["document"]

    t2 = time.perf_counter()
    issues = check(doc)
    t_check = time.perf_counter() - t2

    decision, reasons = decide(doc, issues)

    return Result(
        decision=decision,
        reasons=reasons,
        document=doc,
        issues=issues,
        image={
            "original": prepared.original,
            "sent": prepared.sent,
        },
        model=llm_result.get("model"),
        elapsed={
            "prepare": t_prep,
            "extraction": t_llm,
            "checks": t_check,
        },
        usage=llm_result.get("usage"),
    )
