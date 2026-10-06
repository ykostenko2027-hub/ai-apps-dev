from dataclasses import dataclass, field
import json
import os
import time
from dotenv import load_dotenv
from . import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(question: str, customer_id: str) -> Answer:
    max_rounds = int(os.getenv("TOOL_MAX_ROUNDS", str(MAX_ROUNDS)))
    ctx = tools.Context(customer_id=customer_id)
    messages = llm.build_messages(question)
    tool_specs = tools.specs()

    calls: list[ToolTrace] = []
    total_model_time = 0.0
    total_tools_time = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    model_name = None
    stopped = "answer"
    rounds_done = 0
    final_text = ""

    for current_round in range(1, max_rounds + 1):
        rounds_done = current_round
        chat_res = llm.chat(messages, tool_specs, tool_choice="auto")
        total_model_time += chat_res["elapsed"]
        model_name = chat_res["model"]
        if chat_res["usage"]:
            total_prompt_tokens += chat_res["usage"].get("prompt_tokens", 0)
            total_completion_tokens += chat_res["usage"].get("completion_tokens", 0)

        msg = chat_res["message"]
        messages.append(msg)

        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            final_text = msg.content or ""
            stopped = "answer"
            break

        if current_round == max_rounds:
            for call_item in tool_calls:
                fn_name = call_item.function.name
                fn_args = call_item.function.arguments
                calls.append(
                    ToolTrace(
                        round=current_round,
                        name=fn_name,
                        arguments=fn_args,
                        status="rejected",
                        reason="Перевищено ліміт звертань до моделі",
                        result=None,
                        elapsed=0.0
                    )
                )
            final_text = "Досягнуто ліміту звертань до моделі. Будь ласка, спробуйте уточнити запит."
            stopped = "limit"
            break

        for call_item in tool_calls:
            fn_name = call_item.function.name
            fn_args = call_item.function.arguments

            t_start = time.perf_counter()
            t_result = tools.call(fn_name, fn_args, ctx)
            t_elapsed = time.perf_counter() - t_start
            total_tools_time += t_elapsed

            calls.append(
                ToolTrace(
                    round=current_round,
                    name=fn_name,
                    arguments=fn_args,
                    status=t_result.status,
                    reason=t_result.reason,
                    result=t_result.content,
                    elapsed=round(t_elapsed, 4)
                )
            )

            tool_content_str = (
                json.dumps(t_result.content, ensure_ascii=False)
                if isinstance(t_result.content, (dict, list))
                else str(t_result.content)
            )

            messages.append({
                "role": "tool",
                "tool_call_id": call_item.id,
                "content": tool_content_str
            })

    return Answer(
        text=final_text,
        calls=calls,
        rounds=rounds_done,
        stopped=stopped,
        model=model_name,
        elapsed={
            "model": round(total_model_time, 3),
            "tools": round(total_tools_time, 3)
        },
        usage={
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens
        }
    )
