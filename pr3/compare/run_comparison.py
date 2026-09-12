import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.llm import ask

async def main():
    load_dotenv()
    requests_path = Path(__file__).parent / "requests.json"
    results_path = Path(__file__).parent / "results.json"

    with open(requests_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    requests = data.get("звернення", data.get("zvernennnya", []))
    print(f"=== Початок порівняння {len(requests)} звернень ===", flush=True)

    results = []
    if results_path.exists():
        try:
            with open(results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    model_name = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    config1_temp = 0.2
    config2_temp = 1.0

    completed_indices = {r["index"] for r in results if "config1" in r and "config2" in r}

    for idx, item in enumerate(requests, 1):
        if idx in completed_indices:
            print(f"[{idx}/{len(requests)}] Пропущено (вже пораховано)", flush=True)
            continue

        q = item.get("текст", item.get("text"))
        kind = item.get("вид", item.get("kind", ""))
        print(f"[{idx}/{len(requests)}] ({kind}): {q}", flush=True)

        print("  -> Запит Конфігурація 1 (T=0.2)...", flush=True)
        res1 = await ask(q, temperature=config1_temp, model=model_name)
        print(f"  <- Отримано відповідь 1 ({res1['elapsed']} с)", flush=True)
        await asyncio.sleep(5)

        print("  -> Запит Конфігурація 2 (T=1.0)...", flush=True)
        res2 = await ask(q, temperature=config2_temp, model=model_name)
        print(f"  <- Отримано відповідь 2 ({res2['elapsed']} с)", flush=True)
        await asyncio.sleep(5)

        item_result = {
            "index": idx,
            "kind": kind,
            "question": q,
            "config1": {
                "model": model_name,
                "temperature": config1_temp,
                "answer": res1["answer"],
                "elapsed": res1["elapsed"],
                "length": len(res1["answer"]),
                "tokens": res1.get("tokens", {})
            },
            "config2": {
                "model": model_name,
                "temperature": config2_temp,
                "answer": res2["answer"],
                "elapsed": res2["elapsed"],
                "length": len(res2["answer"]),
                "tokens": res2.get("tokens", {})
            }
        }

        existing_idx = next((i for i, r in enumerate(results) if r["index"] == idx), None)
        if existing_idx is not None:
            results[existing_idx] = item_result
        else:
            results.append(item_result)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    print("=== Порівняння повністю завершено! ===", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
