import json
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import embeddings, index, keyword

def run_evaluation(use_prefix=True):
    idx = index.load()
    kw_idx = keyword.build(idx.chunks)

    queries_path = Path(__file__).parent / "queries.json"
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for q in data["запити"]:
        query_text = q["запит"]
        expected_doc = q.get("очікуваний_документ", "")
        q_filter = q.get("фільтр", None)

        t0 = time.perf_counter()
        if use_prefix:
            q_vec = embeddings.embed_query(query_text)
        else:
            model = embeddings.get_model()
            q_vec = model.encode(query_text, normalize_embeddings=True, convert_to_numpy=True)
        sem_hits = index.search(idx, q_vec, top_k=5, filters=q_filter)
        t_sem = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        kw_hits = keyword.search(kw_idx, query_text, top_k=5, filters=q_filter)
        t_kw = (time.perf_counter() - t0) * 1000

        sem_pos = None
        sem_score = None
        for rank, h in enumerate(sem_hits, start=1):
            if expected_doc and h.chunk.source == expected_doc:
                if sem_pos is None:
                    sem_pos = rank
                    sem_score = h.score

        kw_pos = None
        kw_score = None
        for rank, h in enumerate(kw_hits, start=1):
            if expected_doc and h.chunk.source == expected_doc:
                if kw_pos is None:
                    kw_pos = rank
                    kw_score = h.score

        results.append({
            "kind": q["вид"],
            "query": query_text,
            "expected": expected_doc,
            "filter": q_filter,
            "sem": {
                "pos": sem_pos,
                "score": sem_score,
                "top1_source": sem_hits[0].chunk.source if sem_hits else None,
                "top1_score": sem_hits[0].score if sem_hits else None,
                "time_ms": t_sem
            },
            "kw": {
                "pos": kw_pos,
                "score": kw_score,
                "top1_source": kw_hits[0].chunk.source if kw_hits else None,
                "top1_score": kw_hits[0].score if kw_hits else None,
                "time_ms": t_kw
            }
        })
    return results

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    def report_results(name, results):
        print(f"=== {name} ===")
        bar = "|"
        header = f"{bar} {'Kind':<24} {bar} {'Expected':<36} {bar} {'Sem Pos':<8} {'Sem Score':<10} {bar} {'KW Pos':<8} {'KW Score':<10} {bar}"
        print(header)
        print("-" * len(header))
        s_h1, s_h3, k_h1, k_h3, count = 0, 0, 0, 0, 0
        for r in results:
            exp = r["expected"]
            if not exp:
                top_s = f"{r['sem']['top1_score']:.3f}" if r["sem"]["top1_score"] is not None else "None"
                top_k = f"{r['kw']['top1_score']:.2f}" if r["kw"]["top1_score"] is not None else "None"
                print(f"{bar} {r['kind']:<24} {bar} {'[OUT OF DOMAIN]':<36} {bar} {'-':<8} {top_s:<10} {bar} {'-':<8} {top_k:<10} {bar}")
                continue
            count += 1
            sp = str(r["sem"]["pos"]) if r["sem"]["pos"] else ">5"
            kp = str(r["kw"]["pos"]) if r["kw"]["pos"] else ">5"
            ss = f"{r['sem']['score']:.3f}" if r["sem"]["score"] is not None else "—"
            ks = f"{r['kw']['score']:.2f}" if r["kw"]["score"] is not None else "—"
            if r["sem"]["pos"] == 1: s_h1 += 1
            if r["sem"]["pos"] in (1, 2, 3): s_h3 += 1
            if r["kw"]["pos"] == 1: k_h1 += 1
            if r["kw"]["pos"] in (1, 2, 3): k_h3 += 1
            print(f"{bar} {r['kind']:<24} {bar} {exp:<36} {bar} {sp:<8} {ss:<10} {bar} {kp:<8} {ks:<10} {bar}")
        print("=" * len(header))
        print(f"Semantic: hit@1 = {s_h1}/{count} ({s_h1/count*100:.1f}%), hit@3 = {s_h3}/{count} ({s_h3/count*100:.1f}%)")
        print(f"Keyword:  hit@1 = {k_h1}/{count} ({k_h1/count*100:.1f}%), hit@3 = {k_h3}/{count} ({k_h3/count*100:.1f}%)")
        avg_sem_time = sum(r["sem"]["time_ms"] for r in results) / len(results)
        avg_kw_time = sum(r["kw"]["time_ms"] for r in results) / len(results)
        print(f"Avg Latency: Semantic = {avg_sem_time:.1f} ms, Keyword = {avg_kw_time:.1f} ms")
        print()

    base_results = run_evaluation(use_prefix=True)
    report_results("1. BASELINE (with query: prefix)", base_results)

    exp_results = run_evaluation(use_prefix=False)
    report_results("2. EXPERIMENT (isolated change: without query: prefix)", exp_results)
