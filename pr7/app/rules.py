from dataclasses import dataclass
from decimal import Decimal
import json
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"


@dataclass
class Issue:
    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    return company, suppliers


def _check_iban(iban: str) -> bool:
    iban = iban.replace(" ", "").upper()
    if len(iban) != 29 or not iban.startswith("UA"):
        return False
    if not iban[2:].isdigit():
        return False
    reordered = iban[4:] + iban[:4]
    numeric = "".join(str(ord(ch) - ord("A") + 10) if ch.isalpha() else ch for ch in reordered)
    return int(numeric) % 97 == 1


def _check_edrpou(code: str) -> bool:
    if not (code.isdigit() and len(code) == 8):
        return False
    digits = [int(c) for c in code]
    w1 = [1, 2, 3, 4, 5, 6, 7] if int(code) < 30000000 or int(code) > 60000000 else [7, 1, 2, 3, 4, 5, 6]
    w2 = [3, 4, 5, 6, 7, 8, 9] if int(code) < 30000000 or int(code) > 60000000 else [9, 3, 4, 5, 6, 7, 8]
    s = sum(d * w for d, w in zip(digits[:7], w1))
    rem = s % 11
    if rem < 10:
        return rem == digits[7]
    s2 = sum(d * w for d, w in zip(digits[:7], w2))
    rem2 = s2 % 11
    if rem2 < 10:
        return rem2 == digits[7]
    return digits[7] == 0


def _check_rnokpp(code: str) -> bool:
    if not (code.isdigit() and len(code) == 10):
        return False
    digits = [int(c) for c in code]
    weights = [-1, 5, 7, 9, 4, 6, 10, 5, 7]
    s = sum(d * w for d, w in zip(digits[:9], weights))
    rem = (s % 11) % 10
    return rem == digits[9]


def _check_code(code: str) -> bool:
    if not code:
        return False
    code = code.strip()
    if len(code) == 8:
        return _check_edrpou(code)
    if len(code) == 10:
        return _check_rnokpp(code)
    return False


def check(document: dict) -> list[Issue]:
    issues: list[Issue] = []
    company, suppliers = load_reference()

    doc_type = (document.get("document_type") or "").strip().lower()
    if "рахунок" not in doc_type:
        issues.append(Issue(
            field="document_type",
            rule="document_type",
            message="Тип документа не є рахунком на оплату (видаткова накладна або інший документ)",
            severity="error"
        ))
        return issues

    req_fields = [
        ("document_type", document.get("document_type")),
        ("number", document.get("number")),
        ("date", document.get("date")),
        ("supplier.name", (document.get("supplier") or {}).get("name")),
        ("supplier.code", (document.get("supplier") or {}).get("code")),
        ("supplier.iban", (document.get("supplier") or {}).get("iban")),
        ("buyer.name", (document.get("buyer") or {}).get("name")),
        ("buyer.code", (document.get("buyer") or {}).get("code")),
        ("total_without_vat", document.get("total_without_vat")),
        ("vat", document.get("vat")),
        ("total", document.get("total")),
    ]
    for field_path, val in req_fields:
        if val is None or val == "":
            issues.append(Issue(
                field=field_path,
                rule="missing_required",
                message=f"Обовʼязкове поле '{field_path}' відсутнє або не розпізнано",
                severity="error"
            ))

    items = document.get("items")
    if not isinstance(items, list) or len(items) == 0:
        issues.append(Issue(
            field="items",
            rule="missing_required",
            message="Таблиця позицій відсутня або порожня",
            severity="error"
        ))

    supplier = document.get("supplier") or {}
    s_code = supplier.get("code")
    s_iban = supplier.get("iban")

    if s_code and not _check_code(s_code):
        issues.append(Issue(
            field="supplier.code",
            rule="code_checksum",
            message="Некоректний контрольний розряд коду постачальника (ЄДРПОУ/РНОКПП)",
            severity="error"
        ))

    if s_iban and not _check_iban(s_iban):
        issues.append(Issue(
            field="supplier.iban",
            rule="iban_checksum",
            message="Некоректна контрольна сума IBAN (ISO 13616)",
            severity="error"
        ))

    matching_supp = next((s for s in suppliers if s["code"] == s_code), None)
    if matching_supp:
        if s_iban and matching_supp["iban"].replace(" ", "").upper() != s_iban.replace(" ", "").upper():
            issues.append(Issue(
                field="supplier.iban",
                rule="iban_registry",
                message="IBAN не збігається з довідником постачальників (можлива підміна реквізитів)",
                severity="error"
            ))
    elif s_code:
        issues.append(Issue(
            field="supplier.code",
            rule="supplier_registry",
            message="Постачальник не знайдений у довіднику постачальників",
            severity="error"
        ))

    buyer = document.get("buyer") or {}
    b_code = buyer.get("code")
    b_name = (buyer.get("name") or "").lower()
    if b_code != company["code"] or "сузір" not in b_name:
        issues.append(Issue(
            field="buyer",
            rule="buyer",
            message="Рахунок виставлено іншій компанії, а не покупцю «Сузірʼя Рітейл»",
            severity="error"
        ))

    if items and isinstance(items, list):
        for idx, it in enumerate(items):
            if isinstance(it, dict):
                qty = it.get("quantity")
                price = it.get("price")
                amt = it.get("amount")
                if qty is not None and price is not None and amt is not None:
                    try:
                        dq = Decimal(str(qty))
                        dp = Decimal(str(price))
                        da = Decimal(str(amt))
                        if abs((dq * dp).quantize(Decimal("0.01")) - da) > Decimal("0.01"):
                            issues.append(Issue(
                                field=f"items.{idx}.amount",
                                rule="arithmetic_line",
                                message=f"Сума позиції ({da}) не збігається з кількість × ціна ({dq * dp})",
                                severity="error"
                            ))
                    except Exception:
                        pass

    if document.get("total_without_vat") is not None and items and isinstance(items, list):
        try:
            line_sum = sum(Decimal(str(it["amount"])) for it in items if isinstance(it, dict) and it.get("amount") is not None)
            if abs(line_sum - Decimal(str(document["total_without_vat"]))) > Decimal("0.02"):
                issues.append(Issue(
                    field="total_without_vat",
                    rule="arithmetic_subtotal",
                    message="Сума рядків не збігається з підсумком без ПДВ",
                    severity="error"
                ))
        except Exception:
            pass

    if document.get("total_without_vat") is not None and document.get("vat") is not None:
        try:
            d_sub = Decimal(str(document["total_without_vat"]))
            d_vat = Decimal(str(document["vat"]))
            if matching_supp and not matching_supp.get("vat_payer", True):
                expected_vat = Decimal("0.00")
            elif matching_supp and matching_supp.get("vat_payer", True):
                expected_vat = (d_sub * Decimal("0.20")).quantize(Decimal("0.01"))
            else:
                expected_vat = None
            if expected_vat is not None and abs(d_vat - expected_vat) > Decimal("0.02"):
                issues.append(Issue(
                    field="vat",
                    rule="arithmetic_vat",
                    message="Сума ПДВ не відповідає ставці ПДВ (20% або 0%)",
                    severity="error"
                ))
        except Exception:
            pass

    if document.get("total_without_vat") is not None and document.get("vat") is not None and document.get("total") is not None:
        try:
            d_sub = Decimal(str(document["total_without_vat"]))
            d_vat = Decimal(str(document["vat"]))
            d_tot = Decimal(str(document["total"]))
            if abs((d_sub + d_vat) - d_tot) > Decimal("0.02"):
                issues.append(Issue(
                    field="total",
                    rule="arithmetic_total",
                    message="Сума до сплати не збігається з сумою без ПДВ + ПДВ",
                    severity="error"
                ))
        except Exception:
            pass

    doc_date = document.get("date")
    valid_until = document.get("valid_until")
    if doc_date and valid_until:
        if str(valid_until) < str(doc_date):
            issues.append(Issue(
                field="valid_until",
                rule="date_validity",
                message="Строк дії рахунку раніше дати виставлення",
                severity="error"
            ))

    return issues
