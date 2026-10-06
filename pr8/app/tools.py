from dataclasses import dataclass
import json
import re
import jsonschema
from shop import service


@dataclass
class Context:
    customer_id: str


@dataclass
class ToolResult:
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


def specs() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": "list_orders",
                "description": "Отримати перелік усіх замовлень поточного клієнта з їхніми статусами, датами та сумами.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_order",
                "description": "Отримати детальну інформацію про конкретне замовлення клієнта за його номером: склад товарів, статус, вартість та доставка. Дозволено лише для замовлень поточного клієнта.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Номер замовлення, наприклад '10458'",
                        }
                    },
                    "required": ["order_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_products",
                "description": "Пошук товарів у каталозі магазину за словами в назві, описі чи категорії.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Пошуковий запит (назва або ключові слова)",
                        },
                        "category": {
                            "type": "string",
                            "description": "Категорія товару",
                        },
                        "max_price": {
                            "type": "number",
                            "description": "Максимальна ціна",
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 50,
                            "description": "Максимальна кількість результатів",
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_product",
                "description": "Отримати картку товару за його артикулом (SKU): опис, ціна, вага, гарантія, можливість повернення.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару, наприклад 'ORN-PRO'",
                        }
                    },
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_stock",
                "description": "Перевірити наявність товару на складі за його артикулом (SKU) та дату очікуваного надходження.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару, наприклад 'ORN-PRO'",
                        }
                    },
                    "required": ["sku"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "delivery_quote",
                "description": "Розрахувати вартість і термін доставки товарів у місто України.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Місто доставки в Україні",
                        },
                        "method": {
                            "type": "string",
                            "enum": ["branch", "courier", "pickup"],
                            "description": "Спосіб доставки: branch (відділення), courier (кур'єр), pickup (самовивіз)",
                        },
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sku": {
                                        "type": "string",
                                        "description": "Артикул товару",
                                    },
                                    "quantity": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "description": "Кількість одиниць",
                                    },
                                },
                                "required": ["sku"],
                                "additionalProperties": False,
                            },
                            "minItems": 1,
                            "description": "Список товарів для розрахунку доставки",
                        },
                    },
                    "required": ["city", "method", "items"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_return",
                "description": "Оформити заявку на повернення товару з отриманого замовлення поточного клієнта.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_id": {
                            "type": "string",
                            "description": "Номер замовлення",
                        },
                        "sku": {
                            "type": "string",
                            "description": "Артикул товару для повернення",
                        },
                        "reason": {
                            "type": "string",
                            "enum": ["not_suitable", "defect", "wrong_item", "damaged"],
                            "description": "Причина повернення: not_suitable (не підійшов), defect (дефект), wrong_item (інший товар), damaged (пошкоджено при доставці)",
                        },
                        "quantity": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Кількість",
                        },
                        "comment": {
                            "type": "string",
                            "description": "Коментар клієнта",
                        },
                    },
                    "required": ["order_id", "sku", "reason"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    spec_map = {tool["function"]["name"]: tool["function"] for tool in specs()}
    if name not in spec_map:
        return ToolResult(
            status="rejected",
            reason=f"Невідомий інструмент: {name}",
            content={"rejected": True, "error": f"Інструмент '{name}' не існує або недоступний"}
        )

    try:
        args = json.loads(raw_arguments) if raw_arguments else {}
        if not isinstance(args, dict):
            return ToolResult(
                status="rejected",
                reason="Аргументи мають бути JSON-об'єктом",
                content={"rejected": True, "error": "Аргументи мають бути JSON-об'єктом"}
            )
    except Exception as exc:
        return ToolResult(
            status="rejected",
            reason=f"Некоректний JSON аргументів: {exc}",
            content={"rejected": True, "error": f"Некоректний JSON: {exc}"}
        )

    try:
        jsonschema.validate(instance=args, schema=spec_map[name]["parameters"])
    except jsonschema.ValidationError as exc:
        return ToolResult(
            status="rejected",
            reason=f"Помилка валідації схеми: {exc.message}",
            content={"rejected": True, "error": f"Некоректні параметри: {exc.message}"},
            arguments=args
        )

    try:
        if name == "list_orders":
            raw = service.list_orders(ctx.customer_id)
            return ToolResult(status="ok", content=raw, arguments=args)

        if name == "get_order":
            order_id_clean = re.sub(r"[^\d]", "", str(args.get("order_id", "")))
            if not order_id_clean:
                order_id_clean = str(args.get("order_id", "")).strip()

            if order_id_clean in service._state["orders"]:
                ord_obj = service._state["orders"][order_id_clean]
                if ord_obj["customer_id"] != ctx.customer_id:
                    return ToolResult(
                        status="rejected",
                        reason=f"Замовлення {order_id_clean} належить іншому клієнту",
                        content={"rejected": True, "error": f"Замовлення {order_id_clean} не знайдено серед замовлень вашого облікового запису"},
                        arguments={"order_id": order_id_clean}
                    )

            full_order = service.get_order(order_id_clean)
            if full_order["customer_id"] != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    reason=f"Замовлення {order_id_clean} належить іншому клієнту",
                    content={"rejected": True, "error": f"Замовлення {order_id_clean} не знайдено серед замовлень вашого облікового запису"},
                    arguments={"order_id": order_id_clean}
                )

            filtered_order = {
                "order_id": full_order["order_id"],
                "created_at": full_order["created_at"],
                "status": full_order["status"],
                "status_label": full_order.get("status_label", ""),
                "items": full_order.get("items", []),
                "total": full_order.get("total", ""),
                "delivery": {
                    "method": full_order.get("delivery", {}).get("method"),
                    "carrier": full_order.get("delivery", {}).get("carrier"),
                    "city": full_order.get("delivery", {}).get("city"),
                    "point": full_order.get("delivery", {}).get("point"),
                    "tracking": full_order.get("delivery", {}).get("tracking"),
                    "delivered_at": full_order.get("delivery", {}).get("delivered_at"),
                    "cost": full_order.get("delivery", {}).get("cost"),
                }
            }
            return ToolResult(status="ok", content=filtered_order, arguments={"order_id": order_id_clean})

        if name == "search_products":
            res = service.search_products(
                query=args.get("query", ""),
                category=args.get("category"),
                max_price=args.get("max_price"),
                limit=args.get("limit", 10)
            )
            return ToolResult(status="ok", content=res, arguments=args)

        if name == "get_product":
            p = service.get_product(args["sku"])
            filtered_p = {
                "sku": p["sku"],
                "name": p["name"],
                "category": p["category"],
                "price": p["price"],
                "weight_kg": p["weight_kg"],
                "warranty_months": p["warranty_months"],
                "returnable": p["returnable"],
                "bulky": p["bulky"],
                "description": p["description"],
            }
            return ToolResult(status="ok", content=filtered_p, arguments=args)

        if name == "get_stock":
            s = service.get_stock(args["sku"])
            return ToolResult(status="ok", content=s, arguments=args)

        if name == "delivery_quote":
            q = service.delivery_quote(city=args["city"], method=args["method"], items=args["items"])
            return ToolResult(status="ok", content=q, arguments=args)

        if name == "create_return":
            order_id_clean = re.sub(r"[^\d]", "", str(args.get("order_id", "")))
            if not order_id_clean:
                order_id_clean = str(args.get("order_id", "")).strip()

            if order_id_clean in service._state["orders"]:
                ord_obj = service._state["orders"][order_id_clean]
                if ord_obj["customer_id"] != ctx.customer_id:
                    return ToolResult(
                        status="rejected",
                        reason=f"Замовлення {order_id_clean} належить іншому клієнту",
                        content={"rejected": True, "error": f"Замовлення {order_id_clean} не знайдено серед замовлень вашого облікового запису"},
                        arguments=args
                    )

            ret = service.create_return(
                order_id=order_id_clean,
                sku=args["sku"],
                reason=args["reason"],
                quantity=args.get("quantity", 1),
                comment=args.get("comment", "")
            )
            return ToolResult(status="ok", content=ret, arguments=args)

    except service.NotFound as exc:
        return ToolResult(status="error", reason=str(exc), content={"error": exc.code, "message": str(exc)}, arguments=args)
    except service.PolicyViolation as exc:
        return ToolResult(status="error", reason=str(exc), content={"error": exc.code, "message": str(exc)}, arguments=args)
    except service.InvalidRequest as exc:
        return ToolResult(status="error", reason=str(exc), content={"error": exc.code, "message": str(exc)}, arguments=args)
    except service.Unavailable as exc:
        return ToolResult(status="error", reason=str(exc), content={"error": exc.code, "message": "Сервіс магазину тимчасово недоступний. Спробуйте пізніше."}, arguments=args)
    except service.ShopError as exc:
        return ToolResult(status="error", reason=str(exc), content={"error": exc.code, "message": str(exc)}, arguments=args)
    except Exception as exc:
        return ToolResult(status="error", reason=str(exc), content={"error": "unexpected_error", "message": str(exc)}, arguments=args)

    return ToolResult(status="rejected", reason="Непідтримувана операція", content={"rejected": True, "error": "Непідтримувана операція"}, arguments=args)
