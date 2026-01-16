# eval/run_rag_eval.py

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests


def main():
    base_url = os.getenv("RAG_EVAL_BASE_URL", "http://127.0.0.1:8000")
    questions_path = Path(os.getenv("RAG_EVAL_QUESTIONS", "data/rag_eval_questions.jsonl"))
    out_path = Path("outputs/rag_eval_results.json")
    out_path.parent.mkdir(exist_ok=True)

    if not questions_path.exists():
        raise SystemExit(f"Questions file not found: {questions_path}")

    results = []
    t0 = time.time()
    n = 0
    cited = 0
    not_found = 0

    with questions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q = obj.get("question") or obj.get("q") or ""
            if not q:
                continue

            start = time.time()
            r = requests.post(f"{base_url}/ask", json={"ticker": "AAPL", "question": q, "k": 6}, timeout=60)
            latency = time.time() - start

            data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text}
            ans = (data.get("answer") or "").strip()

            cits = data.get("citations") or []
            if cits:
                cited += 1
            if ans == "Not found in sources.":
                not_found += 1

            results.append(
                {
                    "question": q,
                    "status_code": r.status_code,
                    "latency_s": latency,
                    "answer": ans,
                    "citations": cits,
                }
            )
            n += 1

    total_s = time.time() - t0
    summary = {
        "n": n,
        "citation_coverage_rate": (cited / n) if n else 0.0,
        "not_found_rate": (not_found / n) if n else 0.0,
        "avg_latency_s": (sum(x["latency_s"] for x in results) / n) if n else 0.0,
        "total_runtime_s": total_s,
    }

    out = {"summary": summary, "results": results}
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
