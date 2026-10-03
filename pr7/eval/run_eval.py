import json
import os
from pathlib import Path
import sys
import time
from decimal import Decimal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import extraction
from app import images

EXPECTED_FILE = ROOT / "samples" / "expected.json"
OWN_DIR = ROOT / "samples" / "own"
EVAL_DIR = ROOT / "eval"


def load_all_expected():
    data = json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))
    r02_clean = data["clean/rahunok-02.png"]
    data["own/rahunok-02-degraded.jpg"] = {
        "source": "rahunok-02",
        "variant": "own-degraded",
        "note": "Власний погіршений зразок rahunok-02 (контраст, розмиття, шум)",
        "fields": r02_clean["fields"],
        "expected_checks": [],
        "expected_decision": "auto"
    }
    r04_clean = data["clean/rahunok-04.png"]
    data["own/rahunok-04-phone.jpg"] = {
        "source": "rahunok-04",
        "variant": "own-phone",
        "note": "Власне фото rahunok-04 (нахил, шум, стиснення)",
        "fields": r04_clean["fields"],
        "expected_checks": [],
        "expected_decision": "auto"
    }
    return data


def normalize_val(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return f"{Decimal(str(v)):.2f}"
    if isinstance(v, str):
        s = v.strip().replace("’", "'").replace("ʼ", "'").replace("`", "'")
        try:
            d = Decimal(s.replace(" ", "").replace(",", "."))
            return f"{d:.2f}"
        except Exception:
            return " ".join(s.split())
    return v


def compare_documents(extracted_doc, expected_doc):
    field_labels = {}

    def extract_flat(obj, prefix=""):
        res = {}
        if not isinstance(obj, dict):
            return res
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                res.update(extract_flat(v, path))
            elif isinstance(v, list):
                if k == "items":
                    for idx, it in enumerate(v):
                        if isinstance(it, dict):
                            for ik, iv in it.items():
                                res[f"items.{idx}.{ik}"] = iv
                else:
                    res[path] = v
            else:
                res[path] = v
        return res

    flat_exp = extract_flat(expected_doc)
    flat_ext = extract_flat(extracted_doc)

    all_keys = set(flat_exp.keys()) | set(flat_ext.keys())
    for k in sorted(all_keys):
        exp_v = normalize_val(flat_exp.get(k))
        ext_v = normalize_val(flat_ext.get(k))

        if exp_v is not None and ext_v is not None:
            if exp_v == ext_v:
                label = "правильно"
            else:
                label = "помилка"
        elif exp_v is not None and ext_v is None:
            label = "пропуск"
        elif exp_v is None and ext_v is not None:
            label = "вигадка"
        else:
            label = "правильно"
        field_labels[k] = {
            "expected": exp_v,
            "extracted": ext_v,
            "label": label
        }
    return field_labels


def evaluate_dataset(max_side=1600, out_filename="results_1600px.json", resume=True):
    os.environ["IMAGE_MAX_SIDE"] = str(max_side)
    images.IMAGE_MAX_SIDE = max_side

    dataset = load_all_expected()
    out_path = EVAL_DIR / out_filename
    results = {}
    if resume and out_path.exists():
        try:
            results = json.loads(out_path.read_text(encoding="utf-8"))
        except Exception:
            results = {}

    print(f"Starting evaluation with IMAGE_MAX_SIDE={max_side} (total {len(dataset)} samples)...")

    for idx, (rel_path, item) in enumerate(dataset.items(), start=1):
        if rel_path in results and results[rel_path].get("actual_decision") not in ("error", None):
            print(f"[{idx}/{len(dataset)}] Reusing existing result for {rel_path}")
            continue

        file_path = ROOT / "samples" / rel_path
        if not file_path.exists():
            print(f"[{idx}/{len(dataset)}] Skipping missing file: {rel_path}")
            continue

        print(f"[{idx}/{len(dataset)}] Processing {rel_path}...")
        content = file_path.read_bytes()

        try:
            res = extraction.process(content)
            res_dict = {
                "decision": res.decision,
                "reasons": res.reasons,
                "document": res.document,
                "issues": [
                    {"field": i.field, "rule": i.rule, "message": i.message, "severity": i.severity}
                    for i in res.issues
                ],
                "image": res.image,
                "model": res.model,
                "elapsed": res.elapsed,
                "usage": res.usage
            }
        except Exception as exc:
            print(f"Error on {rel_path}: {exc}")
            res_dict = {
                "decision": "error",
                "reasons": [str(exc)],
                "document": None,
                "issues": [],
                "image": {},
                "model": None,
                "elapsed": {},
                "usage": {}
            }

        comparison = compare_documents(res_dict.get("document") or {}, item.get("fields") or {})

        rules_fired = [i["rule"] for i in res_dict.get("issues", [])]
        expected_rules = item.get("expected_checks", [])

        variant = item.get("variant", "clean")
        if "/" in rel_path:
            folder = rel_path.split("/")[0]
            if folder == "clean":
                variant = "clean"
            elif folder == "own":
                variant = "own"
            else:
                s = rel_path.split("/")[-1]
                variant = s.rsplit(".", 1)[0].split("-")[-1]

        results[rel_path] = {
            "variant": variant,
            "expected_decision": item.get("expected_decision"),
            "actual_decision": res_dict["decision"],
            "expected_checks": expected_rules,
            "actual_checks": rules_fired,
            "issues": res_dict.get("issues", []),
            "reasons": res_dict.get("reasons", []),
            "field_comparison": comparison,
            "elapsed": res_dict.get("elapsed", {}),
            "usage": res_dict.get("usage", {}),
            "document": res_dict.get("document"),
            "image": res_dict.get("image", {})
        }

        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        time.sleep(1.0)

    return results


def summarize(results):
    variants = {}
    critical_fields = {"supplier.iban", "supplier.code", "total"}

    for path, data in results.items():
        v = data["variant"]
        if v not in variants:
            variants[v] = {
                "count": 0,
                "correct": 0,
                "errors": 0,
                "omissions": 0,
                "hallucinations": 0,
                "caught_by_rules": 0,
                "silent_errors": 0,
                "decision_match": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "crit_correct": 0,
                "crit_total": 0,
            }

        var_stat = variants[v]
        var_stat["count"] += 1

        fc = data["field_comparison"]
        has_error = False
        has_crit_error = False

        for fk, finfo in fc.items():
            lbl = finfo["label"]
            is_crit = any(fk == cf or fk.startswith(cf + ".") for cf in critical_fields)
            if is_crit:
                var_stat["crit_total"] += 1
                if lbl == "правильно":
                    var_stat["crit_correct"] += 1
                else:
                    has_crit_error = True

            if lbl == "правильно":
                var_stat["correct"] += 1
            elif lbl == "помилка":
                var_stat["errors"] += 1
                has_error = True
            elif lbl == "пропуск":
                var_stat["omissions"] += 1
                has_error = True
            elif lbl == "вигадка":
                var_stat["hallucinations"] += 1
                has_error = True

        if len(data["actual_checks"]) > 0:
            var_stat["caught_by_rules"] += 1

        if data["actual_decision"] == "auto" and (has_error or has_crit_error):
            var_stat["silent_errors"] += 1

        if data["actual_decision"] == data["expected_decision"]:
            var_stat["decision_match"] += 1

        u = data.get("usage") or {}
        var_stat["prompt_tokens"] += u.get("prompt_tokens", 0)
        var_stat["completion_tokens"] += u.get("completion_tokens", 0)

    return variants


def main():
    res1600 = evaluate_dataset(max_side=1600, out_filename="results_1600px.json", resume=True)
    summary1600 = summarize(res1600)
    (EVAL_DIR / "summary_1600px.json").write_text(json.dumps(summary1600, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Run 1 (1600px) completed.")

    res800 = evaluate_dataset(max_side=800, out_filename="results_800px.json", resume=True)
    summary800 = summarize(res800)
    (EVAL_DIR / "summary_800px.json").write_text(json.dumps(summary800, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Run 2 (800px) completed.")


if __name__ == "__main__":
    main()
