"""Інструменти взаємодії з сервісом магазину «Сузірʼя».

Використовуються як агентом із ПР8, так і workflow із ПР9.
Виклик іде через call(name, raw_arguments, ctx) і повертає ToolResult.
Клієнт визначається виключно через ctx.customer_id, що унеможливлює
доступ до чужих замовлень. Службові поля (notes) фільтруються перед
передачею моделі.
"""

from dataclasses import dataclass
import json
import copy

from shop import service


@dataclass
class Context:
    """Контекст сеансу клієнта."""
    customer_id: str


@dataclass
class ToolResult:
    """Результат виклику інструмента."""
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "Отримати перелік усіх замовлень поточного клієнта (номер, дата, статус, сума).",
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
            "description": "Отримати детальну інформацію про замовлення за його номером.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення, наприклад '10412'.",
                    },
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
            "description": "Пошук товарів у каталозі магазину за словами в назві або категорії.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Пошуковий запит (назва або тип товару).",
                    },
                    "category": {
                        "type": "string",
                        "description": "Категорія товару.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Максимальна кількість товарів у відповіді (від 1 до 50).",
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
            "description": "Отримати детальну інформацію про товар (картку товару) за артикулом (SKU).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару (SKU), наприклад 'SIR-R10'.",
                    },
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
            "description": "Перевірити наявність товару на складі за артикулом (SKU).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару (SKU).",
                    },
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
            "description": "Розрахувати вартість і строки доставки товарів у вказане місто.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Місто доставки.",
                    },
                    "method": {
                        "type": "string",
                        "enum": ["branch", "courier", "pickup"],
                        "description": "Спосіб доставки: branch (відділення), courier (курʼєр), pickup (самовивіз).",
                    },
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "quantity": {"type": "integer"},
                            },
                            "required": ["sku"],
                        },
                        "description": "Список товарів і їх кількості.",
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
            "description": "Оформити заявку на повернення товару із замовлення клієнта.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення.",
                    },
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару, який повертається.",
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["not_suitable", "defect", "wrong_item", "damaged"],
                        "description": "Причина повернення.",
                    },
                    "quantity": {
                        "type": "integer",
                        "description": "Кількість товару, що повертається (за замовчуванням 1).",
                    },
                    "comment": {
                        "type": "string",
                        "description": "Коментар клієнта.",
                    },
                },
                "required": ["order_id", "sku", "reason"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "Скасувати замовлення (можливо лише у статусі 'Готується').",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення.",
                    },
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_returns",
            "description": "Переглянути список уже оформлених повернень замовлення.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]


def specs() -> list[dict]:
    """Повернути описи інструментів у форматі OpenAI tools."""
    return copy.deepcopy(TOOLS_SCHEMA)


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    """Виконати виклик інструмента з валідацією прав доступу."""
    try:
        args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else (raw_arguments or {})
    except Exception as exc:
        return ToolResult(
            status="rejected",
            reason=f"Некоректний JSON аргументів: {exc}",
            arguments={"raw": str(raw_arguments)},
        )

    try:
        if name == "list_orders":
            res = service.list_orders(ctx.customer_id)
            return ToolResult(status="ok", content=res, arguments=args)

        elif name == "get_order":
            order_id = str(args.get("order_id", "")).strip()
            if not order_id:
                return ToolResult(status="rejected", reason="Не вказано order_id", arguments=args)
            try:
                order = service.get_order(order_id)
            except service.NotFound:
                return ToolResult(status="rejected", reason=f"Замовлення {order_id} не знайдено", arguments=args, content=f"Замовлення {order_id} не знайдено.")
            
            # Перевірка власника
            if str(order.get("customer_id")) != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    reason=f"Замовлення {order_id} не належить поточному клієнту",
                    arguments=args,
                    content=f"Замовлення {order_id} не знайдено серед ваших замовлень.",
                )
            # Фільтрація службових полів (notes)
            filtered = copy.deepcopy(order)
            filtered.pop("notes", None)
            return ToolResult(status="ok", content=filtered, arguments=args)

        elif name == "search_products":
            query = args.get("query", "")
            category = args.get("category")
            limit = int(args.get("limit", 10))
            res = service.search_products(query=query, category=category, limit=limit)
            return ToolResult(status="ok", content=res, arguments=args)

        elif name == "get_product":
            sku = str(args.get("sku", "")).strip()
            if not sku:
                return ToolResult(status="rejected", reason="Не вказано sku", arguments=args)
            res = service.get_product(sku)
            return ToolResult(status="ok", content=res, arguments=args)

        elif name == "get_stock":
            sku = str(args.get("sku", "")).strip()
            if not sku:
                return ToolResult(status="rejected", reason="Не вказано sku", arguments=args)
            res = service.get_stock(sku)
            return ToolResult(status="ok", content=res, arguments=args)

        elif name == "delivery_quote":
            city = args.get("city", "")
            method = args.get("method", "branch")
            items = args.get("items", [])
            res = service.delivery_quote(city=city, method=method, items=items)
            return ToolResult(status="ok", content=res, arguments=args)

        elif name == "create_return":
            order_id = str(args.get("order_id", "")).strip()
            sku = str(args.get("sku", "")).strip()
            reason = str(args.get("reason", "")).strip()
            quantity = int(args.get("quantity", 1))
            comment = str(args.get("comment", "")).strip()

            # Перевірка належності замовлення
            order = service.get_order(order_id)
            if str(order.get("customer_id")) != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    reason=f"Замовлення {order_id} не належить поточному клієнту",
                    arguments=args,
                    content=f"Замовлення {order_id} не знайдено серед ваших замовлень.",
                )

            res = service.create_return(
                order_id=order_id,
                sku=sku,
                reason=reason,
                quantity=quantity,
                comment=comment,
            )
            return ToolResult(status="ok", content=res, arguments=args)

        elif name == "cancel_order":
            order_id = str(args.get("order_id", "")).strip()
            # Перевірка належності замовлення
            order = service.get_order(order_id)
            if str(order.get("customer_id")) != ctx.customer_id:
                return ToolResult(
                    status="rejected",
                    reason=f"Замовлення {order_id} не належить поточному клієнту",
                    arguments=args,
                    content=f"Замовлення {order_id} не знайдено серед ваших замовлень.",
                )

            res = service.cancel_order(order_id=order_id)
            filtered = copy.deepcopy(res)
            filtered.pop("notes", None)
            return ToolResult(status="ok", content=filtered, arguments=args)

        elif name == "list_returns":
            order_id = args.get("order_id")
            if order_id:
                order_id = str(order_id).strip()
                order = service.get_order(order_id)
                if str(order.get("customer_id")) != ctx.customer_id:
                    return ToolResult(
                        status="rejected",
                        reason=f"Замовлення {order_id} не належить поточному клієнту",
                        arguments=args,
                        content=[],
                    )
                res = service.list_returns(order_id=order_id)
            else:
                # Отримати тільки повернення для замовлень даного клієнта
                customer_orders = {o["order_id"] for o in service.list_orders(ctx.customer_id)}
                all_returns = service.list_returns()
                res = [r for r in all_returns if str(r.get("order_id")) in customer_orders]

            return ToolResult(status="ok", content=res, arguments=args)

        else:
            return ToolResult(
                status="rejected",
                reason=f"Невідомий інструмент: {name}",
                arguments=args,
                content=f"Інструмент {name} не підтримується.",
            )

    except service.NotFound as exc:
        return ToolResult(status="rejected", reason=str(exc), arguments=args, content=str(exc))
    except (service.PolicyViolation, service.InvalidRequest) as exc:
        return ToolResult(status="rejected", reason=str(exc), arguments=args, content=str(exc))
    except service.Unavailable as exc:
        return ToolResult(status="error", reason="Сервіс магазину тимчасово недоступний", arguments=args, content="Сервіс магазину тимчасово недоступний")
    except Exception as exc:
        return ToolResult(status="error", reason=str(exc), arguments=args, content=str(exc))
