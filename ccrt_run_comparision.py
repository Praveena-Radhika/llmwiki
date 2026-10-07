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

AZURE_ENDPOINT = "https://ease-azure-ai.openai.azure.com/"
AZURE_API_KEY = "YOUR_ACTUAL_API_KEY"
AZURE_API_VERSION = "2024-12-01-preview"
MODEL = "gpt-5.4"

RAG_DB = "./chroma_store"
RAG_COLLECTION = "extracted-doc"

WIKI_DB = "./wiki_chroma_store"

BENCHMARK_FILE = "evaluation/benchmark.json"

OUTPUT_DIR = Path("evaluation/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Number of benchmark questions
MAX_QUESTIONS = 50

# Increasing query-load levels
FREQUENCY_LEVELS = [1, 2, 4, 8]

# Keep low because Azure previously returned 429
MAX_WORKERS = 2

# Retrieval settings
RAG_TOP_K = 30
WIKI_TOP_K = 5

# Retry settings
MAX_RETRIES = 7
BASE_BACKOFF = 3


# ============================================================
# CLIENTS
# ============================================================

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
)

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-mpnet-base-v2")
print("Embedding model loaded.")


# ============================================================
# LOAD BENCHMARK
# ============================================================

with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

# Handle different possible benchmark.json structures
if isinstance(benchmark, dict):

    if "questions" in benchmark:
        benchmark = benchmark["questions"]

    elif "items" in benchmark:
        benchmark = benchmark["items"]

    elif "benchmark" in benchmark:
        benchmark = benchmark["benchmark"]

    else:
        benchmark = list(benchmark.values())

if not isinstance(benchmark, list):
    raise ValueError(
        "benchmark.json must contain a list of benchmark questions."
    )

benchmark = benchmark[:MAX_QUESTIONS]

print(f"Loaded {len(benchmark)} benchmark questions")


# ============================================================
# CHROMA
# ============================================================

print("Loading RAG database...")

rag_db = chromadb.PersistentClient(path=RAG_DB)

rag_collection = rag_db.get_collection(
    RAG_COLLECTION
)

print(
    f"RAG collection loaded: "
    f"{rag_collection.count()} vectors"
)


print("Loading Wiki database...")

wiki_db = chromadb.PersistentClient(
    path=WIKI_DB
)


def find_wiki_collection():

    collections = wiki_db.list_collections()

    if not collections:
        raise RuntimeError(
            "No collections found in wiki_chroma_store. "
            "Run build_wiki_index.py first."
        )

    candidates = []

    for collection in collections:

        try:
            count = collection.count()
        except Exception:
            count = 0

        name = collection.name.lower()

        if (
            "wiki" in name
            or "index" in name
            or "concept" in name
        ):
            candidates.append(
                (count, collection)
            )

    if not candidates:

        candidates = [
            (
                collection.count(),
                collection
            )
            for collection in collections
        ]

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected = candidates[0][1]

    print(
        f"Wiki collection loaded: "
        f"{selected.name} "
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
# RAG RETRIEVAL
# ============================================================

def retrieve_rag(question):

    vector = embed(question)

    result = rag_collection.query(
        query_embeddings=[vector],
        n_results=RAG_TOP_K,
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = result.get(
        "documents",
        [[]]
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]]
    )[0]

    contexts = []

    for doc, meta in zip(
        documents,
        metadatas
    ):

        contexts.append(
            {
                "text": doc,
                "metadata": meta
            }
        )

    return contexts


# ============================================================
# WIKI RETRIEVAL
# ============================================================

def retrieve_wiki(question):

    vector = embed(question)

    result = wiki_collection.query(
        query_embeddings=[vector],
        n_results=WIKI_TOP_K,
        include=[
            "documents",
            "metadatas"
        ]
    )

    documents = result.get(
        "documents",
        [[]]
    )[0]

    metadatas = result.get(
        "metadatas",
        [[]]
    )[0]

    contexts = []

    for doc, meta in zip(
        documents,
        metadatas
    ):

        contexts.append(
            {
                "text": doc,
                "metadata": meta
            }
        )

    return contexts


# ============================================================
# ANSWER PROMPT
# ============================================================

def answer_prompt(
    question,
    contexts,
    system_name
):

    context_text = "\n\n".join(
        f"[SOURCE {i + 1}]\n{c['text']}"
        for i, c in enumerate(contexts)
    )

    return f"""
You are evaluating a knowledge retrieval system for
Rolls-Royce SMR engineering and regulatory documents.

SYSTEM:
{system_name}

QUESTION:
{question}

SOURCE MATERIAL:
{context_text}

Instructions:

1. Answer ONLY using the supplied source material.
2. Do not use outside knowledge.
3. If the source does not contain enough information, say exactly:
   "Insufficient information in the source documents."
4. Be concise but complete.
5. Do not invent values, relationships, dates or technical details.

Answer:
"""


# ============================================================
# OPENAI GENERATION
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

            latency = (
                time.perf_counter()
                - start
            )

            answer = (
                response.choices[0]
                .message.content
            )

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
                prompt_tokens
                + completion_tokens
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

            error_text = (
                last_error.lower()
            )

            is_rate_limit = (
                "429" in error_text
                or "rate_limit" in error_text
                or "too many requests"
                in error_text
            )

            if not is_rate_limit:
                break

            wait = (
                BASE_BACKOFF
                * (2 ** attempt)
            )

            print(
                f"429 rate limit. "
                f"Retry "
                f"{attempt + 1}/{MAX_RETRIES} "
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
# RUN ONE QUESTION
# ============================================================

def run_one(
    item,
    system_name
):

    question = item["question"]

    if system_name == "RAG":
        contexts = retrieve_rag(
            question
        )
    else:
        contexts = retrieve_wiki(
            question
        )

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
        "prompt_tokens": result[
            "prompt_tokens"
        ],
        "completion_tokens": result[
            "completion_tokens"
        ],
        "total_tokens": result[
            "total_tokens"
        ],
        "latency": result[
            "latency"
        ],
        "success": result[
            "success"
        ],
        "error": result[
            "error"
        ]
    }


# ============================================================
# QUALITY JUDGE
# ============================================================

JUDGE_PROMPT = """
You are a strict evaluator comparing an AI answer against
a reference answer for Rolls-Royce SMR technical documentation.

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
Does it directly answer the question?

Hallucination resistance:
Does it avoid unsupported claims?

Technical correctness:
Is the technical interpretation correct?

For unanswerable questions, the answer should correctly abstain.

Return ONLY valid JSON:

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
        question=record[
            "question"
        ],
        reference=record[
            "reference_answer"
        ],
        answer=record[
            "answer"
        ]
    )

    result = generate(prompt)

    if not result["success"]:

        return {
            **record,
            "judge_success": False,
            "judge_error": result[
                "error"
            ]
        }

    text = result[
        "answer"
    ].strip()

    try:

        score = json.loads(text)

        quality_values = [
            score[
                "factual_accuracy"
            ],
            score[
                "semantic_similarity"
            ],
            score[
                "completeness"
            ],
            score[
                "relevance"
            ],
            score[
                "hallucination_resistance"
            ],
            score[
                "technical_correctness"
            ]
        ]

        quality_score = (
            sum(quality_values)
            / len(quality_values)
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
# PHASE 1: QUALITY
# ============================================================

def run_quality_evaluation():

    print("\n")
    print("=" * 70)
    print("PHASE 1 — ANSWER QUALITY")
    print("=" * 70)

    all_results = []

    for system_name in [
        "RAG",
        "WikiLLM"
    ]:

        print(
            f"\nRunning {system_name}..."
        )

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
                for i, item
                in enumerate(benchmark)
            }

            for completed, future in enumerate(
                as_completed(futures),
                1
            ):

                try:

                    result = (
                        future.result()
                    )

                    results.append(
                        result
                    )

                except Exception as e:

                    print(
                        f"Query failed: {e}"
                    )

                print(
                    f"{system_name}: "
                    f"{completed}/"
                    f"{len(benchmark)}"
                )

        # Judge each answer once
        print(
            f"\nJudging {system_name}..."
        )

        judged = []

        for i, result in enumerate(
            results,
            1
        ):

            if result["success"]:

                judged.append(
                    judge_one(result)
                )

            else:

                judged.append(
                    result
                )

            print(
                f"Judge {system_name}: "
                f"{i}/{len(results)}"
            )

        all_results.extend(
            judged
        )

    output = (
        OUTPUT_DIR
        / "quality_results.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

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
# PHASE 2: FREQUENCY / LOAD
# ============================================================

def run_frequency_test():

    print("\n")
    print("=" * 70)
    print("PHASE 2 — INCREASING QUERY FREQUENCY")
    print("=" * 70)

    results = []

    for multiplier in FREQUENCY_LEVELS:

        print("\n")
        print("-" * 70)
        print(
            f"FREQUENCY LEVEL: {multiplier}x"
        )
        print("-" * 70)

        workload = (
            benchmark * multiplier
        )

        for system_name in [
            "RAG",
            "WikiLLM"
        ]:

            print(
                f"\n{system_name}: "
                f"{len(workload)} queries"
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

                        completed_results.append(
                            {
                                "success": False,
                                "error": str(e),
                                "latency": None,
                                "total_tokens": 0
                            }
                        )

                    if (
                        i % 10 == 0
                        or i == len(workload)
                    ):

                        print(
                            f"{system_name}: "
                            f"{i}/"
                            f"{len(workload)}"
                        )

            elapsed = (
                time.perf_counter()
                - start
            )

            successful = [
                r
                for r in completed_results
                if r.get("success")
            ]

            latencies = [
                r["latency"]
                for r in successful
                if r.get("latency")
                is not None
            ]

            total_tokens = sum(
                r.get(
                    "total_tokens",
                    0
                )
                for r
                in completed_results
            )

            success_rate = (
                len(successful)
                / len(completed_results)
                * 100
            )

            avg_latency = (
                statistics.mean(
                    latencies
                )
                if latencies
                else None
            )

            p95_latency = None

            if latencies:

                sorted_latencies = sorted(
                    latencies
                )

                index = min(
                    len(
                        sorted_latencies
                    ) - 1,
                    int(
                        len(
                            sorted_latencies
                        ) * 0.95
                    )
                )

                p95_latency = (
                    sorted_latencies[index]
                )

            throughput = (
                len(successful)
                / elapsed
                if elapsed > 0
                else 0
            )

            results.append(
                {
                    "frequency_multiplier":
                        multiplier,

                    "system":
                        system_name,

                    "queries":
                        len(workload),

                    "successful_queries":
                        len(successful),

                    "failed_queries":
                        (
                            len(completed_results)
                            - len(successful)
                        ),

                    "success_rate":
                        success_rate,

                    "elapsed_seconds":
                        elapsed,

                    "throughput_qps":
                        throughput,

                    "avg_latency_seconds":
                        avg_latency,

                    "p95_latency_seconds":
                        p95_latency,

                    "total_tokens":
                        total_tokens,

                    "avg_tokens_per_query":
                        (
                            total_tokens
                            / len(
                                completed_results
                            )
                            if completed_results
                            else 0
                        )
                }
            )

    output = (
        OUTPUT_DIR
        / "frequency_results.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

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
# FINAL VERDICT
# ============================================================

def create_verdict(
    quality_results,
    frequency_results
):

    print("\n")
    print("=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    systems = [
        "RAG",
        "WikiLLM"
    ]

    quality_summary = {}

    # --------------------------------------------------------
    # QUALITY SUMMARY
    # --------------------------------------------------------

    for system in systems:

        rows = [
            r
            for r in quality_results
            if (
                r.get("system")
                == system
                and r.get(
                    "judge_success"
                )
            )
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

        semantic = statistics.mean(
            r["semantic_similarity"]
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
            "factual_accuracy": factual,
            "semantic_similarity": semantic,
            "completeness": completeness,
            "relevance": relevance,
            "hallucination_resistance":
                hallucination,
            "technical_correctness":
                technical
        }

    print("\nQUALITY SCORE")

    for system, values in (
        quality_summary.items()
    ):

        print(
            f"{system}: "
            f"{values['quality']:.2f}/10"
        )

    # --------------------------------------------------------
    # FREQUENCY EFFICIENCY
    # --------------------------------------------------------

    frequency_winners = []

    print("\nFREQUENCY / EFFICIENCY")

    for multiplier in (
        FREQUENCY_LEVELS
    ):

        rows = [
            r
            for r in frequency_results
            if r[
                "frequency_multiplier"
            ] == multiplier
        ]

        if len(rows) < 2:
            continue

        rag = next(
            r
            for r in rows
            if r["system"] == "RAG"
        )

        wiki = next(
            r
            for r in rows
            if r["system"] == "WikiLLM"
        )

        systems_rows = [
            rag,
            wiki
        ]

        max_throughput = max(
            r["throughput_qps"]
            for r in systems_rows
        )

        max_latency = max(
            r["avg_latency_seconds"]
            or 1
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
                r["success_rate"]
                / 100
            )

            latency_score = (
                1
                - (
                    (
                        r[
                            "avg_latency_seconds"
                        ]
                        or max_latency
                    )
                    / max_latency
                )
                if max_latency
                else 0
            )

            token_score = (
                1
                - (
                    r[
                        "avg_tokens_per_query"
                    ]
                    / max_tokens
                )
                if max_tokens
                else 0
            )

            r[
                "efficiency_score"
            ] = (
                0.35
                * throughput_score
                + 0.30
                * reliability_score
                + 0.20
                * max(
                    0,
                    latency_score
                )
                + 0.15
                * max(
                    0,
                    token_score
                )
            )

        winner = max(
            systems_rows,
            key=lambda x:
                x["efficiency_score"]
        )["system"]

        frequency_winners.append(
            {
                "frequency":
                    multiplier,

                "winner":
                    winner,

                "RAG_efficiency":
                    rag[
                        "efficiency_score"
                    ],

                "WikiLLM_efficiency":
                    wiki[
                        "efficiency_score"
                    ],

                "RAG_success_rate":
                    rag[
                        "success_rate"
                    ],

                "WikiLLM_success_rate":
                    wiki[
                        "success_rate"
                    ],

                "RAG_avg_latency":
                    rag[
                        "avg_latency_seconds"
                    ],

                "WikiLLM_avg_latency":
                    wiki[
                        "avg_latency_seconds"
                    ],

                "RAG_avg_tokens":
                    rag[
                        "avg_tokens_per_query"
                    ],

                "WikiLLM_avg_tokens":
                    wiki[
                        "avg_tokens_per_query"
                    ]
            }
        )

        print(
            f"{multiplier}x load → {winner}"
        )

    # --------------------------------------------------------
    # QUALITY WINNER
    # --------------------------------------------------------

    quality_winner = None

    if len(quality_summary) == 2:

        quality_winner = max(
            quality_summary,
            key=lambda system:
                quality_summary[
                    system
                ]["quality"]
        )

    # --------------------------------------------------------
    # FREQUENCY WINNER
    # --------------------------------------------------------

    efficiency_counts = {
        "RAG": 0,
        "WikiLLM": 0
    }

    for row in frequency_winners:

        efficiency_counts[
            row["winner"]
        ] += 1

    frequency_winner = max(
        efficiency_counts,
        key=efficiency_counts.get
    )

    # --------------------------------------------------------
    # FIND BREAKPOINT
    # --------------------------------------------------------

    breakpoint_frequency = None

    for row in frequency_winners:

        if (
            row["winner"]
            == "WikiLLM"
        ):

            breakpoint_frequency = (
                row["frequency"]
            )

            break

    # --------------------------------------------------------
    # FINAL VERDICT
    # --------------------------------------------------------

    if (
        quality_winner
        == frequency_winner
    ):

        verdict = (
            f"{quality_winner} is the overall "
            f"stronger system for this SMR benchmark, "
            f"combining answer quality with performance "
            f"under increasing query frequency."
        )

    else:

        if breakpoint_frequency:

            verdict = (
                f"RAG and WikiLLM show a trade-off: "
                f"{quality_winner} provides the stronger "
                f"answer quality, while {frequency_winner} "
                f"is more efficient under increasing "
                f"query load. WikiLLM becomes the "
                f"frequency winner at {breakpoint_frequency}x "
                f"in this experiment."
            )

        else:

            verdict = (
                f"{quality_winner} provides the stronger "
                f"answer quality, while "
                f"{frequency_winner} provides better "
                f"operational efficiency under the "
                f"tested query loads."
            )

    print("\n")
    print("=" * 70)
    print(
        f"Quality winner: {quality_winner}"
    )
    print(
        f"Frequency/efficiency winner: "
        f"{frequency_winner}"
    )

    if breakpoint_frequency:
        print(
            f"WikiLLM efficiency breakpoint: "
            f"{breakpoint_frequency}x"
        )

    print("\nVERDICT:")
    print(verdict)
    print("=" * 70)

    # --------------------------------------------------------
    # SAVE FINAL RESULT
    # --------------------------------------------------------

    final_result = {
        "benchmark_questions":
            len(benchmark),

        "frequency_levels":
            FREQUENCY_LEVELS,

        "quality_summary":
            quality_summary,

        "frequency_results":
            frequency_winners,

        "quality_winner":
            quality_winner,

        "frequency_efficiency_winner":
            frequency_winner,

        "wiki_frequency_breakpoint":
            breakpoint_frequency,

        "verdict":
            verdict
    }

    output = (
        OUTPUT_DIR
        / "final_verdict.json"
    )

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            final_result,
            f,
            indent=2
        )

    print(
        f"\nFinal verdict saved to:"
        f"\n{output}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 70)
    print("RAG vs WikiLLM — SMR EVALUATION")
    print("=" * 70)

    quality_results = (
        run_quality_evaluation()
    )

    frequency_results = (
        run_frequency_test()
    )

    create_verdict(
        quality_results,
        frequency_results
    )

    print("\nDONE.")