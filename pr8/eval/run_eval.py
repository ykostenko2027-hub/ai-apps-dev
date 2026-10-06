import os
import sys
import json
import time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shop import service
from app import assistant


def run_test_suite(output_file: str, max_rounds: int = 3, failure_rate: float = 0.0):
    os.environ["TOOL_MAX_ROUNDS"] = str(max_rounds)
    os.environ["SHOP_FAILURE_RATE"] = str(failure_rate)

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    questions_path = os.path.join(eval_dir, "questions.json")

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []

    for item in questions:
        service.reset()
        q_id = item["id"]
        customer = item["customer"]
        question = item["question"]

        start_time = time.perf_counter()
        try:
            ans = assistant.answer(question, customer)
            elapsed = time.perf_counter() - start_time
            record = {
                "id": q_id,
                "kind": item.get("kind"),
                "customer": customer,
                "question": question,
                "expect_tools": item.get("expect_tools"),
                "expect": item.get("expect"),
                "answer": ans.text,
                "rounds": ans.rounds,
                "stopped": ans.stopped,
                "calls": [asdict(c) for c in ans.calls],
                "elapsed": ans.elapsed,
                "usage": ans.usage,
                "total_time": round(elapsed, 3),
                "error": None
            }
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            record = {
                "id": q_id,
                "kind": item.get("kind"),
                "customer": customer,
                "question": question,
                "expect_tools": item.get("expect_tools"),
                "expect": item.get("expect"),
                "answer": None,
                "rounds": 0,
                "stopped": "crash",
                "calls": [],
                "elapsed": {},
                "usage": {},
                "total_time": round(elapsed, 3),
                "error": str(exc)
            }

        results.append(record)
        print(f"[{q_id}] rounds: {record['rounds']}, calls: {len(record['calls'])}, stopped: {record['stopped']}")
        time.sleep(5)

    out_path = os.path.join(eval_dir, output_file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Results written to {out_path}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "base"
    if mode == "base":
        run_test_suite("results_base.json", max_rounds=3, failure_rate=0.0)
    elif mode == "rounds2":
        run_test_suite("results_rounds2.json", max_rounds=2, failure_rate=0.0)
    elif mode == "failure":
        run_test_suite("results_failure.json", max_rounds=3, failure_rate=0.3)
