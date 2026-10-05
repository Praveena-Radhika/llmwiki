import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import AzureOpenAI, RateLimitError

INPUT = Path("evaluation/raw_results.json")
OUTPUT = Path("evaluation/final_results.json")
CSV_OUTPUT = Path("evaluation/summary.csv")

AZURE_ENDPOINT = "https://ease-azure-ai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_API_KEY = "PASTE_YOUR_API_KEY_HERE"
WORKERS = 6

METRICS = [
    "factual_correctness",
    "semantic_similarity",
    "completeness",
    "relevance",
    "hallucination_avoidance",
    "technical_accuracy",
]


def client():
    if AZURE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise SystemExit("Put your Azure API key in AZURE_API_KEY first.")
    return AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )


def judge_one(client_obj, record, retries=4):
    prompt = f"""
You are the independent evaluator for a research comparison between Traditional RAG and a
concept-centric WikiLLM system over Rolls-Royce SMR engineering/regulatory documents.

Evaluate the two answers independently against the reference answer and the question.
Do NOT reward an answer merely because it is longer.
Do NOT use outside knowledge.

QUESTION:
{record['question']}

REFERENCE ANSWER:
{record['reference_answer']}

RAG ANSWER:
{record['rag']['answer']}

WIKILLM ANSWER:
{record['wikillm']['answer']}

Score each metric from 0 to 10:
- factual_correctness: factual agreement with the reference answer
- semantic_similarity: whether the answer expresses the same meaning, even with different wording
- completeness: whether all important parts of the reference answer are covered
- relevance: whether it directly answers the question without unnecessary material
- hallucination_avoidance: absence of unsupported, invented, or contradictory claims
- technical_accuracy: correctness of technical terminology and relationships relative to the reference

For an unanswerable question, a correct refusal such as "Insufficient information..." should score very high;
an invented specific answer should score very low, especially for factual correctness and hallucination avoidance.

Return ONLY JSON:
{{
  "rag": {{
    "factual_correctness": 0,
    "semantic_similarity": 0,
    "completeness": 0,
    "relevance": 0,
    "hallucination_avoidance": 0,
    "technical_accuracy": 0,
    "reason": "short reason"
  }},
  "wikillm": {{
    "factual_correctness": 0,
    "semantic_similarity": 0,
    "completeness": 0,
    "relevance": 0,
    "hallucination_avoidance": 0,
    "technical_accuracy": 0,
    "reason": "short reason"
  }}
}}
"""
    last = None
    for attempt in range(retries):
        try:
            try:
                r = client_obj.chat.completions.create(
                    model=AZURE_DEPLOYMENT,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=900,
                )
            except Exception as e:
                if "response_format" in str(e).lower() or "unsupported" in str(e).lower() or "400" in str(e):
                    r = client_obj.chat.completions.create(
                        model=AZURE_DEPLOYMENT,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=900,
                    )
                else:
                    raise
            raw = r.choices[0].message.content or "{}"
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                data = json.loads(m.group(0), strict=False) if m else None
            if not data or "rag" not in data or "wikillm" not in data:
                raise ValueError("Invalid judge JSON")
            return data
        except RateLimitError as e:
            last = e
            time.sleep(min(5 * (attempt + 1), 30))
        except Exception as e:
            last = e
            time.sleep(2)
    raise last


def clean_scores(block):
    out = {}
    for metric in METRICS:
        try:
            value = float(block.get(metric, 0))
        except Exception:
            value = 0.0
        out[metric] = max(0.0, min(10.0, value))
    out["overall"] = round(sum(out[m] for m in METRICS) / len(METRICS), 3)
    out["reason"] = str(block.get("reason", "")).strip()
    return out


def main():
    if not INPUT.exists():
        raise SystemExit("evaluation/raw_results.json not found. Run run_comparison.py first.")
    if OUTPUT.exists():
        raise SystemExit("evaluation/final_results.json already exists. Delete it only to intentionally rerun judging.")

    data = json.loads(INPUT.read_text(encoding="utf-8"))
    records = data["records"]
    print(f"Judging {len(records)} questions with {WORKERS} workers...")

    results = [None] * len(records)

    def worker(i, record):
        judged = judge_one(client(), record)
        return i, {
            **record,
            "scores": {
                "rag": clean_scores(judged["rag"]),
                "wikillm": clean_scores(judged["wikillm"]),
            },
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(worker, i, r) for i, r in enumerate(records)]
        for n, fut in enumerate(as_completed(futures), 1):
            i, result = fut.result()
            results[i] = result
            print(f"Judged {n}/{len(records)}")

    summary = {}
    for system in ("rag", "wikillm"):
        summary[system] = {}
        for metric in METRICS + ["overall"]:
            vals = [r["scores"][system][metric] for r in results]
            summary[system][metric] = round(sum(vals) / len(vals), 3)
        summary[system]["avg_total_tokens"] = round(
            sum(r[system]["total_tokens"] for r in results) / len(results), 2
        )

    delta = {}
    for metric in METRICS + ["overall"]:
        delta[metric] = round(summary["rag"][metric] - summary["wikillm"][metric], 3)
    delta["avg_total_tokens"] = round(
        summary["rag"]["avg_total_tokens"] - summary["wikillm"]["avg_total_tokens"], 2
    )

    winners = {}
    for metric in METRICS + ["overall"]:
        a = summary["rag"][metric]
        b = summary["wikillm"][metric]
        winners[metric] = "RAG" if a > b else "WikiLLM" if b > a else "Tie"

    output = {
        "benchmark_size": len(results),
        "metrics_scale": "0-10",
        "summary": summary,
        "delta_rag_minus_wikillm": delta,
        "winner": winners,
        "records": results,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    with CSV_OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "RAG", "WikiLLM", "RAG_minus_WikiLLM", "winner"])
        for metric in METRICS + ["overall", "avg_total_tokens"]:
            a = summary["rag"][metric]
            b = summary["wikillm"][metric]
            d = round(a - b, 3 if metric != "avg_total_tokens" else 2)
            writer.writerow([metric, a, b, d, winners.get(metric, "Lower is better for tokens")])

    print("\n========================================")
    print("FINAL COMPARISON")
    print("========================================")
    for metric in METRICS + ["overall"]:
        print(f"{metric:24s} RAG={summary['rag'][metric]:5.2f}  WikiLLM={summary['wikillm'][metric]:5.2f}  Winner={winners[metric]}")
    print(f"{'avg_total_tokens':24s} RAG={summary['rag']['avg_total_tokens']:8.2f}  WikiLLM={summary['wikillm']['avg_total_tokens']:8.2f}")
    print(f"\nSaved: {OUTPUT}")
    print(f"Saved: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()
