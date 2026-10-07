import os
import json
import time
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI


# ============================================================
# CONFIG
# ============================================================

AZURE_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://ease-azure-ai.openai.azure.com/"
)

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

AZURE_API_VERSION = "2024-12-01-preview"
MODEL = "gpt-5.4"

# Your existing stores
RAG_DB = "./chroma_store"
RAG_COLLECTION = "extracted-doc"

WIKI_DB = "./wiki_chroma_store"

BENCHMARK_FILE = "evaluation/benchmark.json"

OUTPUT_DIR = Path("evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Same benchmark questions for both systems
MAX_QUESTIONS = 50

# Frequency/load levels.
# 1x = normal
# 2x = twice the workload
# 4x = four times
# 8x = eight times
FREQUENCY_LEVELS = [1, 2, 4, 8]

# IMPORTANT:
# Keep this low because Azure already rate-limited your previous run.
MAX_WORKERS = 2

# Retrieval sizes aligned with the reference repo
RAG_TOP_K = 30
WIKI_TOP_K = 5

# Number of related wiki pages to expand
WIKI_RELATED = 10

# Retry settings for 429
MAX_RETRIES = 7
BASE_BACKOFF = 3


# ============================================================
# CLIENTS
# ============================================================

if not AZURE_API_KEY:
    raise RuntimeError(
        "AZURE_OPENAI_API_KEY environment variable is not set."
    )

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

embedding_model = SentenceTransformer("all-mpnet-base-v2")


# ============================================================
# LOAD BENCHMARK
# ============================================================

with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

if isinstance(benchmark, dict):
    if "questions" in benchmark:
        benchmark = benchmark["questions"]
    elif "items" in benchmark:
        benchmark = benchmark["items"]

benchmark = benchmark[:MAX_QUESTIONS]

print(f"Loaded {len(benchmark)} benchmark questions")


# ============================================================
# CHROMA
# ============================================================

rag_db = chromadb.PersistentClient(path=RAG_DB)
rag_collection = rag_db.get_collection(RAG_COLLECTION)

wiki_db = chromadb.PersistentClient(path=WIKI_DB)


def find_wiki_collection():
    """
    Automatically find the largest likely Wiki collection.
    This avoids hard-coding the collection name created by
    build_wiki_index.py.
    """

    collections = wiki_db.list_collections()

    if not collections:
        raise RuntimeError(
            "No collections found in wiki_chroma_store. "
            "Run build_wiki_index.py first."
        )

    candidates = []

    for c in collections:
        try:
            count = c.count()
        except Exception:
            count = 0

        name = c.name.lower()

        if "wiki" in name or "index" in name or "concept" in name:
            candidates.append((count, c))

    if not candidates:
        candidates = [
            (c.count(), c)
            for c in collections
        ]

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = candidates[0][1]

    print(
        f"Wiki collection: {selected.name} "
        f"({selected.count()} entries)"
    )

    return selected


wiki_collection = find_wiki_collection()


# ============================================================
# EMBEDDING
# ============================================================

def embed(text):
    return embedding_model.encode(
        [text],
        normalize_embeddings=True
    )[0].tolist()


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_rag(question):

    vector = embed(question)

    result = rag_collection.query(
        query_embeddings=[vector],
        n_results=RAG_TOP_K,
        include=["documents", "metadatas"]
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    contexts = []

    for doc, meta in zip(documents, metadatas):
        contexts.append({
            "text": doc,
            "metadata": meta
        })

    return contexts


def retrieve_wiki(question):

    vector = embed(question)

    result = wiki_collection.query(
        query_embeddings=[vector],
        n_results=WIKI_TOP_K,
        include=["documents", "metadatas"]
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    contexts = []

    for doc, meta in zip(documents, metadatas):
        contexts.append({
            "text": doc,
            "metadata": meta
        })

    return contexts


# ============================================================
# PROMPTS
# ============================================================

def answer_prompt(question, contexts, system_name):

    context_text = "\n\n".join(
        f"[SOURCE {i+1}]\n{c['text']}"
        for i, c in enumerate(contexts)
    )

    return f"""
You are evaluating a knowledge retrieval system for Rolls-Royce SMR
engineering and regulatory documents.

SYSTEM:
{system_name}

QUESTION:
{question}

SOURCE MATERIAL:
{context_text}

Instructions:

1. Answer ONLY using the supplied source material.
2. Do not use outside knowledge.
3. If the source does not contain enough information, explicitly say:
   "Insufficient information in the source documents."
4. Be concise but complete.
5. Do not invent values, relationships, dates or technical details.

Answer:
"""


# ============================================================
# OPENAI CALL WITH RATE-LIMIT HANDLING
# ============================================================

def generate(prompt):

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            start = time.perf_counter()

            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
            )

            latency = time.perf_counter() - start

            answer = response.choices[0].message.content

            usage = response.usage

            prompt_tokens = getattr(
                usage,
                "prompt_tokens",
                0
            )

            completion_tokens = getattr(
                usage,
                "completion_tokens",
                0
            )

            total_tokens = getattr(
                usage,
                "total_tokens",
                prompt_tokens + completion_tokens
            )

            return {
                "answer": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency": latency,
                "success": True,
                "error": None
            }

        except Exception as e:

            last_error = str(e)

            error_text = last_error.lower()

            is_rate_limit = (
                "429" in error_text
                or "rate_limit" in error_text
                or "too many requests" in error_text
            )

            if not is_rate_limit:
                break

            wait = BASE_BACKOFF * (2 ** attempt)

            print(
                f"429 rate limit. "
                f"Retry {attempt + 1}/{MAX_RETRIES} "
                f"after {wait}s"
            )

            time.sleep(wait)

    return {
        "answer": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency": None,
        "success": False,
        "error": last_error
    }


# ============================================================
# SINGLE QUERY
# ============================================================

def run_one(item, system_name):

    question = item["question"]

    if system_name == "RAG":
        contexts = retrieve_rag(question)
    else:
        contexts = retrieve_wiki(question)

    prompt = answer_prompt(
        question,
        contexts,
        system_name
    )

    result = generate(prompt)

    return {
        "question": question,
        "reference_answer": item.get(
            "reference_answer",
            ""
        ),
        "question_type": item.get(
            "question_type",
            ""
        ),
        "system": system_name,
        "answer": result["answer"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "latency": result["latency"],
        "success": result["success"],
        "error": result["error"],
    }


# ============================================================
# QUALITY JUDGE
# ============================================================

JUDGE_PROMPT = """
You are a strict evaluator comparing an AI answer against a reference
answer for Rolls-Royce SMR technical documentation.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

SYSTEM ANSWER:
{answer}

Score the answer from 0 to 10 for each criterion.

Factual accuracy:
Does the answer state correct facts?

Semantic similarity:
Does it convey the same meaning as the reference?

Completeness:
Does it cover the important information?

Relevance:
Does it directly answer the question without unnecessary material?

Hallucination resistance:
Does it avoid unsupported claims?

Technical correctness:
Is the technical interpretation correct?

For unanswerable questions, the answer should correctly abstain.

Return ONLY JSON:

{{
  "factual_accuracy": 0,
  "semantic_similarity": 0,
  "completeness": 0,
  "relevance": 0,
  "hallucination_resistance": 0,
  "technical_correctness": 0,
  "reason": "brief reason"
}}
"""


def judge_one(record):

    prompt = JUDGE_PROMPT.format(
        question=record["question"],
        reference=record["reference_answer"],
        answer=record["answer"]
    )

    result = generate(prompt)

    if not result["success"]:
        return {
            **record,
            "judge_success": False,
            "judge_error": result["error"]
        }

    text = result["answer"].strip()

    try:
        score = json.loads(text)

        quality_values = [
            score["factual_accuracy"],
            score["semantic_similarity"],
            score["completeness"],
            score["relevance"],
            score["hallucination_resistance"],
            score["technical_correctness"],
        ]

        quality_score = sum(quality_values) / len(
            quality_values
        )

        return {
            **record,
            **score,
            "quality_score": quality_score,
            "judge_success": True,
            "judge_error": None
        }

    except Exception as e:

        return {
            **record,
            "judge_success": False,
            "judge_error": str(e)
        }


# ============================================================
# RUN QUALITY BASELINE
# ============================================================

def run_quality_evaluation():

    print("\n" + "=" * 70)
    print("PHASE 1 — QUALITY EVALUATION")
    print("=" * 70)

    all_results = []

    for system_name in ["RAG", "WikiLLM"]:

        print(f"\nRunning {system_name} quality evaluation...")

        results = []

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = {
                executor.submit(
                    run_one,
                    item,
                    system_name
                ): i
                for i, item in enumerate(benchmark)
            }

            for completed, future in enumerate(
                as_completed(futures),
                1
            ):

                try:
                    result = future.result()
                    results.append(result)

                except Exception as e:
                    print(
                        f"Query failed: {e}"
                    )

                print(
                    f"{system_name}: "
                    f"{completed}/{len(benchmark)}"
                )

        # Judge only ONCE per question.
        # This prevents frequency testing from
        # artificially multiplying judge cost.
        print(f"Judging {system_name}...")

        judged = []

        for i, result in enumerate(results, 1):

            if result["success"]:
                judged.append(
                    judge_one(result)
                )
            else:
                judged.append(result)

            print(
                f"Judge {system_name}: "
                f"{i}/{len(results)}"
            )

        all_results.extend(judged)

    output = OUTPUT_DIR / "quality_results.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            all_results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved: {output}"
    )

    return all_results


# ============================================================
# FREQUENCY / LOAD EXPERIMENT
# ============================================================

def run_frequency_test():

    print("\n" + "=" * 70)
    print("PHASE 2 — FREQUENCY / LOAD EVALUATION")
    print("=" * 70)

    results = []

    # We use the same 50 questions.
    #
    # At 1x  -> 50 requests
    # At 2x  -> 100 requests
    # At 4x  -> 200 requests
    # At 8x  -> 400 requests
    #
    # Quality is NOT judged again.
    # We measure operational behaviour:
    # latency, tokens, throughput, failures.

    for multiplier in FREQUENCY_LEVELS:

        print(
            f"\n{'-' * 60}\n"
            f"FREQUENCY LEVEL: {multiplier}x\n"
            f"{'-' * 60}"
        )

        workload = (
            benchmark * multiplier
        )

        for system_name in ["RAG", "WikiLLM"]:

            print(
                f"\nRunning {system_name} "
                f"with {len(workload)} queries..."
            )

            start = time.perf_counter()

            completed_results = []

            with ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                futures = [
                    executor.submit(
                        run_one,
                        item,
                        system_name
                    )
                    for item in workload
                ]

                for i, future in enumerate(
                    as_completed(futures),
                    1
                ):

                    try:
                        completed_results.append(
                            future.result()
                        )

                    except Exception as e:

                        completed_results.append({
                            "success": False,
                            "error": str(e),
                            "latency": None,
                            "total_tokens": 0
                        })

                    if i % 10 == 0 or i == len(workload):
                        print(
                            f"{system_name}: "
                            f"{i}/{len(workload)}"
                        )

            elapsed = time.perf_counter() - start

            successful = [
                r for r in completed_results
                if r.get("success")
            ]

            latencies = [
                r["latency"]
                for r in successful
                if r.get("latency") is not None
            ]

            total_tokens = sum(
                r.get("total_tokens", 0)
                for r in completed_results
            )

            success_rate = (
                len(successful)
                / len(completed_results)
                * 100
            )

            avg_latency = (
                statistics.mean(latencies)
                if latencies
                else None
            )

            p95_latency = None

            if latencies:
                sorted_latencies = sorted(latencies)
                index = min(
                    len(sorted_latencies) - 1,
                    int(len(sorted_latencies) * 0.95)
                )
                p95_latency = sorted_latencies[index]

            throughput = (
                len(successful) / elapsed
                if elapsed > 0
                else 0
            )

            results.append({
                "frequency_multiplier": multiplier,
                "system": system_name,
                "queries": len(workload),
                "successful_queries": len(successful),
                "failed_queries": (
                    len(completed_results)
                    - len(successful)
                ),
                "success_rate": success_rate,
                "elapsed_seconds": elapsed,
                "throughput_qps": throughput,
                "avg_latency_seconds": avg_latency,
                "p95_latency_seconds": p95_latency,
                "total_tokens": total_tokens,
                "avg_tokens_per_query": (
                    total_tokens
                    / len(completed_results)
                    if completed_results
                    else 0
                )
            })

    output = OUTPUT_DIR / "frequency_results.json"

    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2
        )

    print(
        f"\nSaved: {output}"
    )

    return results


# ============================================================
# VERDICT
# ============================================================

def create_verdict(quality_results, frequency_results):

    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    systems = ["RAG", "WikiLLM"]

    quality_summary = {}

    for system in systems:

        rows = [
            r for r in quality_results
            if r.get("system") == system
            and r.get("judge_success")
        ]

        if not rows:
            continue

        quality = statistics.mean(
            r["quality_score"]
            for r in rows
        )

        factual = statistics.mean(
            r["factual_accuracy"]
            for r in rows
        )

        completeness = statistics.mean(
            r["completeness"]
            for r in rows
        )

        relevance = statistics.mean(
            r["relevance"]
            for r in rows
        )

        hallucination = statistics.mean(
            r["hallucination_resistance"]
            for r in rows
        )

        technical = statistics.mean(
            r["technical_correctness"]
            for r in rows
        )

        quality_summary[system] = {
            "quality": quality,
            "factual": factual,
            "completeness": completeness,
            "relevance": relevance,
            "hallucination": hallucination,
            "technical": technical
        }

    print("\nQUALITY")

    for system, values in quality_summary.items():

        print(
            f"{system}: "
            f"{values['quality']:.2f}/10"
        )

    # --------------------------------------------------------
    # Frequency winner
    # --------------------------------------------------------

    frequency_winners = []

    print("\nFREQUENCY RESULTS")

    for multiplier in FREQUENCY_LEVELS:

        rows = [
            r for r in frequency_results
            if r["frequency_multiplier"] == multiplier
        ]

        if len(rows) < 2:
            continue

        rag = next(
            r for r in rows
            if r["system"] == "RAG"
        )

        wiki = next(
            r for r in rows
            if r["system"] == "WikiLLM"
        )

        # Efficiency score:
        # higher throughput
        # higher reliability
        # lower latency
        # lower tokens
        #
        # We normalize within each frequency.

        systems_rows = [rag, wiki]

        max_throughput = max(
            r["throughput_qps"]
            for r in systems_rows
        )

        max_latency = max(
            r["avg_latency_seconds"] or 1
            for r in systems_rows
        )

        max_tokens = max(
            r["avg_tokens_per_query"]
            for r in systems_rows
        )

        for r in systems_rows:

            throughput_score = (
                r["throughput_qps"]
                / max_throughput
                if max_throughput
                else 0
            )

            reliability_score = (
                r["success_rate"] / 100
            )

            latency_score = (
                1
                - (
                    (r["avg_latency_seconds"] or max_latency)
                    / max_latency
                )
                if max_latency
                else 0
            )

            token_score = (
                1
                - (
                    r["avg_tokens_per_query"]
                    / max_tokens
                )
                if max_tokens
                else 0
            )

            r["efficiency_score"] = (
                0.35 * throughput_score
                + 0.30 * reliability_score
                + 0.20 * max(0, latency_score)
                + 0.15 * max(0, token_score)
            )

        winner = max(
            systems_rows,
            key=lambda x: x["efficiency_score"]
        )["system"]

        frequency_winners.append({
            "frequency": multiplier,
            "winner": winner,
            "RAG_efficiency": rag["efficiency_score"],
            "WikiLLM_efficiency": wiki["efficiency_score"]
        })

        print(
            f"{multiplier}x load → {winner}"
        )

    # --------------------------------------------------------
    # Overall decision
    # --------------------------------------------------------

    quality_winner = None

    if len(quality_summary) == 2:

        quality_winner = max(
            quality_summary,
            key=lambda x:
                quality_summary[x]["quality"]
        )

    efficiency_counts = {
        "RAG": 0,
        "WikiLLM": 0
    }

    for row in frequency_winners:
        efficiency_counts[row["winner"]] += 1

    frequency_winner = max(
        efficiency_counts,
        key=efficiency_counts.get
    )

    print("\n" + "=" * 70)

    print(
        f"Quality winner: {quality_winner}"
    )

    print(
        f"Frequency/efficiency winner: "
        f"{frequency_winner}"
    )

    # --------------------------------------------------------
    # Human-readable conclusion
    # --------------------------------------------------------

    if quality_winner == frequency_winner:

        final_winner = quality_winner

        verdict = (
            f"{final_winner} provides the strongest overall result "
            f"for this SMR benchmark, considering both answer quality "
            f"and behaviour as query frequency increases."
        )

    else:

        verdict = (
            f"The results show a trade-off: {quality_winner} "
            f"has the stronger answer quality, while {frequency_winner} "
            f"is more efficient/reliable as query frequency increases. "
            f"Therefore the choice depends on whether answer quality "
            f"or high-frequency operational efficiency is the priority."
        )

    print("\n" + verdict)

    output = OUTPUT_DIR / "final_verdict.json"

    final = {
        "quality_summary": quality_summary,
        "frequency_winners": frequency_winners,
        "verdict": verdict
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(
            final,
            f,
            indent=2
        )

    print(
        f"\nSaved: {output}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("RAG vs WikiLLM — SMR EVALUATION")
    print("=" * 70)

    quality_results = run_quality_evaluation()

    frequency_results = run_frequency_test()

    create_verdict(
        quality_results,
        frequency_results
    )

    print("\nDONE.")