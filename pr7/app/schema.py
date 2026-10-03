import json
import re
from decimal import Decimal
import jsonschema


def output_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "document_type": {"type": ["string", "null"]},
            "number": {"type": ["string", "null"]},
            "date": {"type": ["string", "null"]},
            "valid_until": {"type": ["string", "null"]},
            "supplier": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"]},
                    "iban": {"type": ["string", "null"]},
                },
                "required": ["name", "code", "iban"],
                "additionalProperties": False,
            },
            "buyer": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"]},
                },
                "required": ["name", "code"],
                "additionalProperties": False,
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "unit": {"type": ["string", "null"]},
                        "quantity": {"type": ["number", "null"]},
                        "price": {"type": ["string", "number", "null"]},
                        "amount": {"type": ["string", "number", "null"]},
                    },
                    "required": ["name", "unit", "quantity", "price", "amount"],
                    "additionalProperties": False,
                },
            },
            "total_without_vat": {"type": ["string", "number", "null"]},
            "vat": {"type": ["string", "number", "null"]},
            "total": {"type": ["string", "number", "null"]},
        },
        "required": [
            "document_type",
            "number",
            "date",
            "valid_until",
            "supplier",
            "buyer",
            "items",
            "total_without_vat",
            "vat",
            "total",
        ],
        "additionalProperties": False,
    }


def _norm_amount(val) -> str | None:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return f"{Decimal(str(val)):.2f}"
    if isinstance(val, str):
        v = val.strip().replace("\u00a0", "").replace(" ", "").replace("грн", "").replace("UAH", "").replace(",", ".")
        if not v or v.lower() == "null":
            return None
        try:
            return f"{Decimal(v):.2f}"
        except Exception:
            return val.strip()
    return None


def _norm_date(val) -> str | None:
    if not val or not isinstance(val, str):
        return None
    v = val.strip()
    if v.lower() == "null":
        return None
    m_iso = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", v)
    if m_iso:
        return v
    m_dot = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", v)
    if m_dot:
        d, m, y = m_dot.groups()
        return f"{y}-{int(m):02d}-{int(d):02d}"
    months = {
        "січня": "01", "лютого": "02", "березня": "03", "квітня": "04",
        "травня": "05", "червня": "06", "липня": "07", "серпня": "08",
        "вересня": "09", "жовтня": "10", "листопада": "11", "грудня": "12"
    }
    m_word = re.search(r"(\d{1,2})\s+([а-яіїєґ]+)\s+(\d{4})", v, re.IGNORECASE)
    if m_word:
        d, mon, y = m_word.groups()
        mon_lower = mon.lower()
        if mon_lower in months:
            return f"{y}-{months[mon_lower]}-{int(d):02d}"
    return v


def validate(raw: str) -> dict:
    if not raw or not raw.strip():
        raise ValueError("Порожня відповідь моделі.")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Відповідь моделі не є валідним JSON: {exc}")
    try:
        jsonschema.validate(instance=data, schema=output_schema())
    except jsonschema.ValidationError as exc:
        raise ValueError(f"Відповідь моделі не відповідає схемі: {exc.message}")

    data["date"] = _norm_date(data.get("date"))
    data["valid_until"] = _norm_date(data.get("valid_until"))
    data["total_without_vat"] = _norm_amount(data.get("total_without_vat"))
    data["vat"] = _norm_amount(data.get("vat"))
    data["total"] = _norm_amount(data.get("total"))

    if isinstance(data.get("supplier"), dict):
        if data["supplier"].get("iban"):
            data["supplier"]["iban"] = data["supplier"]["iban"].replace(" ", "").upper()
        if data["supplier"].get("code"):
            data["supplier"]["code"] = data["supplier"]["code"].strip()

    if isinstance(data.get("buyer"), dict):
        if data["buyer"].get("code"):
            data["buyer"]["code"] = data["buyer"]["code"].strip()

    if isinstance(data.get("items"), list):
        for it in data["items"]:
            if isinstance(it, dict):
                it["price"] = _norm_amount(it.get("price"))
                it["amount"] = _norm_amount(it.get("amount"))

    return data
