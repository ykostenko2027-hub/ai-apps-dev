import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path, since: str | None = None) -> list[dict]:
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})", file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def group(records: list[dict], key: str) -> dict:
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)
    return groups


def summary(records: list[dict]) -> None:
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records]
    print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def slow(records: list[dict]) -> None:
    attempts = [r for r in records if r.get("event") == "llm.attempt" and not r.get("cached")]
    if not attempts:
        print("викликів до провайдера немає")
        return

    durations = [r["duration_ms"] for r in attempts if "duration_ms" in r]
    print(f"всього викликів: {len(attempts)}")
    print(f"тривалість (мс): медіана = {percentile(durations, 0.5):.1f}, p95 = {percentile(durations, 0.95):.1f}, макс = {max(durations):.1f}")

    sorted_attempts = sorted(attempts, key=lambda x: x.get("duration_ms", 0), reverse=True)
    print("\nнайповільніші виклики:")
    for r in sorted_attempts[:10]:
        q = r.get("wait_queue_ms", 0)
        f = r.get("first_token_ms", 0)
        d = r.get("duration_ms", 0)
        gen = max(0.0, d - f)
        print(f"  {r.get('draft_id')} | {r.get('provider')} ({r.get('model')}) | спроба {r.get('attempt')} | "
              f"разом {d:.1f} мс (черга: {q:.1f} мс, до 1-го токена: {f:.1f} мс, генерація: {gen:.1f} мс)")

    by_prov = group(attempts, "provider")
    print("\nрозподіл за провайдером:")
    for prov, items in by_prov.items():
        prov_durs = [x["duration_ms"] for x in items if "duration_ms" in x]
        print(f"  {prov}: викликів {len(items)}, медіана = {percentile(prov_durs, 0.5):.1f} мс, p95 = {percentile(prov_durs, 0.95):.1f} мс")


def failed(records: list[dict]) -> None:
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    failed_attempts = [r for r in attempts if r.get("status") == "failed"]
    kinds = Counter(r.get("kind") for r in failed_attempts)

    print(f"невдалих спроб звертань: {len(failed_attempts)} з {len(attempts)}")
    print("збої за видом:")
    for k, cnt in kinds.most_common():
        print(f"  {k}: {cnt}")

    draft_attempts = group(attempts, "draft_id")
    recovered = 0
    initially_failed = 0
    for d_id, atts in draft_attempts.items():
        if not d_id:
            continue
        sorted_atts = sorted(atts, key=lambda x: x.get("attempt", 1))
        if sorted_atts and sorted_atts[0].get("status") == "failed":
            initially_failed += 1
            if any(x.get("status") == "done" for x in sorted_atts):
                recovered += 1

    rec_rate = (recovered / initially_failed * 100) if initially_failed else 0.0
    print(f"\nчернеток із збоєм на першій спробі: {initially_failed}")
    print(f"з них врятовано повторами: {recovered} ({rec_rate:.1f}%)")

    wasted_cost = sum(r.get("cost", 0.0) or 0.0 for r in failed_attempts)
    print(f"вартість невдалих спроб: {wasted_cost:.6f} $")


def repeated(records: list[dict]) -> None:
    created = [r for r in records if r.get("event") == "draft.created"]
    req_counts = Counter((r.get("sample_id") or r.get("request_id")) for r in created)

    print("повторні звернення:")
    for req_id, cnt in req_counts.most_common(10):
        print(f"  {req_id}: замовлено {cnt} раз(ів)")

    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    draft_atts = group(attempts, "draft_id")
    multi_attempts = {k: v for k, v in draft_atts.items() if len(v) > 1 and k}
    print(f"\nчернеток з кількома спробами: {len(multi_attempts)}")
    for d_id, atts in list(multi_attempts.items())[:5]:
        print(f"  {d_id}: {len(atts)} спроб")

    clients = Counter(r.get("client") for r in created)
    print("\nактивність клієнтів:")
    for cl, cnt in clients.most_common(5):
        print(f"  {cl}: {cnt} чернеток")


def spend(records: list[dict]) -> None:
    attempts = [r for r in records if r.get("event") == "llm.attempt" and not r.get("cached")]
    total_cost = sum(r.get("cost", 0.0) or 0.0 for r in attempts)
    tokens_in = sum((r.get("tokens") or {}).get("prompt_tokens", 0) for r in attempts)
    tokens_out = sum((r.get("tokens") or {}).get("completion_tokens", 0) for r in attempts)
    estimated_count = sum(1 for r in attempts if r.get("cost_estimated"))

    print(f"загальні витрати: {total_cost:.6f} $")
    print(f"токени: вхідні {tokens_in}, вихідні {tokens_out}, разом {tokens_in + tokens_out}")
    print(f"записів з оцінкою вартості: {estimated_count} з {len(attempts)}")

    by_model = group(attempts, "model")
    print("\nвитрати за моделями:")
    for m, items in by_model.items():
        m_cost = sum(x.get("cost", 0.0) or 0.0 for x in items)
        print(f"  {m}: {len(items)} викликів, {m_cost:.6f} $")

    failed_cost = sum(r.get("cost", 0.0) or 0.0 for r in attempts if r.get("status") == "failed")
    fallback_cost = sum(r.get("cost", 0.0) or 0.0 for r in attempts if r.get("fallback"))
    print(f"\nвитрати на невдалі спроби: {failed_cost:.6f} $")
    print(f"витрати на запасні варіанти: {fallback_cost:.6f} $")


def cache_report(records: list[dict]) -> None:
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    if not attempts:
        print("записів звертань немає")
        return

    hits = sum(1 for r in attempts if r.get("cached"))
    total = len(attempts)
    rate = (hits / total * 100) if total else 0.0

    print(f"всього запитів: {total}")
    print(f"влучань у кеш: {hits} ({rate:.1f}%)")
    print(f"звертань до провайдера: {total - hits}")


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="брати записи від цієї дати/часу (ISO), наприклад 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        try:
            SECTIONS[name](records)
        except Exception as exc:
            print(f"(помилка: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
