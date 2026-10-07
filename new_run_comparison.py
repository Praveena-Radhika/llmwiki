
import os
import json
import time
import math
import random
import statistics
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Border, Side, Alignment
    from openpyxl.chart import BarChart, LineChart, Reference
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.table import Table, TableStyleInfo
except ImportError:
    raise SystemExit("Install Excel support first: pip install openpyxl")


# ============================================================
# 1. CONFIGURATION
# ============================================================

AZURE_ENDPOINT = "https://ease-azure-ai.openai.azure.com/"
AZURE_API_KEY = "YOUR_ACTUAL_API_KEY"
AZURE_API_VERSION = "2024-12-01-preview"
MODEL = "gpt-5.4"

PROJECT_ROOT = Path(".")
SOURCE_DIR = PROJECT_ROOT / "extracted_doc"
RAG_DB = PROJECT_ROOT / "chroma_store"
RAG_COLLECTION = "extracted-doc"
WIKI_DB = PROJECT_ROOT / "wiki_chroma_store"

BENCHMARK_FILE = PROJECT_ROOT / "evaluation" / "benchmark_100.json"
RESULT_DIR = PROJECT_ROOT / "evaluation" / "results"
EXCEL_FILE = PROJECT_ROOT / "evaluation" / "RAG_vs_WikiLLM_Professional_Report.xlsx"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_QUESTIONS = 100
MAX_WORKERS = 2
RAG_TOP_K = 30
WIKI_TOP_K = 5

# Document-volume experiment on YOUR 34-document corpus.
DOCUMENT_LEVELS = [5, 10, 20, 34]

# Query-frequency experiment.
FREQUENCY_LEVELS = [1, 2, 4, 8]

# 100-question distribution.
QUESTION_TYPES = {
    "Single Fact": 15,
    "Definition / Concept": 15,
    "Technical Explanation": 15,
    "Multi-Document Reasoning": 15,
    "Relationship Understanding": 10,
    "Safety / Regulatory": 10,
    "Comparison": 10,
    "Unanswerable / Abstention": 10,
}

MAX_RETRIES = 7
BASE_BACKOFF = 3
RANDOM_SEED = 42

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# 2. AZURE CLIENT + EMBEDDING MODEL
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
# 3. COMMON LLM CALL
# ============================================================

def llm_call(prompt, temperature=0):
    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            start = time.perf_counter()

            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )

            latency = time.perf_counter() - start
            answer = response.choices[0].message.content or ""
            usage = response.usage

            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(
                usage, "total_tokens",
                prompt_tokens + completion_tokens
            )

            return {
                "text": answer,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency": latency,
                "success": True,
                "error": None,
            }

        except Exception as e:
            last_error = str(e)
            s = last_error.lower()

            if not any(x in s for x in ["429", "rate_limit", "too many requests"]):
                break

            wait = BASE_BACKOFF * (2 ** attempt)
            print(f"429 rate limit; retry {attempt + 1}/{MAX_RETRIES} after {wait}s")
            time.sleep(wait)

    return {
        "text": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "latency": None,
        "success": False,
        "error": last_error,
    }


# ============================================================
# 4. SOURCE DOCUMENTS
# ============================================================

def load_source_documents():
    files = sorted(SOURCE_DIR.glob("*.txt"))

    if not files:
        raise FileNotFoundError(
            f"No .txt files found in {SOURCE_DIR.resolve()}"
        )

    docs = {}
    for path in files:
        docs[path.name] = path.read_text(
            encoding="utf-8",
            errors="ignore"
        )

    print(f"Source documents found: {len(docs)}")
    return docs


source_docs = load_source_documents()
SOURCE_NAMES = list(source_docs.keys())


# ============================================================
# 5. CHUNKING FOR BENCHMARK GENERATION
# ============================================================

def chunk_text(text, chunk_size=3500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_source_chunks():
    rows = []
    for filename, text in source_docs.items():
        for idx, chunk in enumerate(chunk_text(text)):
            rows.append({
                "source_file": filename,
                "chunk_index": idx,
                "text": chunk,
            })
    return rows


source_chunks = build_source_chunks()
print(f"Source chunks available: {len(source_chunks)}")


# ============================================================
# 6. CREATE A VALID 100-QUESTION BENCHMARK
# ============================================================

def benchmark_is_valid(path):
    if not path.exists():
        return False

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) < TARGET_QUESTIONS:
            return False

        required = {"question", "reference_answer", "question_type"}
        return all(
            isinstance(x, dict) and required.issubset(x.keys())
            for x in data[:TARGET_QUESTIONS]
        )
    except Exception:
        return False


QUESTION_GENERATION_PROMPT = """
You are creating a controlled evaluation benchmark for comparing
Traditional RAG against a WikiLLM concept-based retrieval system for
Rolls-Royce SMR engineering/regulatory documents.

Create EXACTLY 4 questions from the supplied evidence.

The four questions must use these types:
1. Single Fact
2. Definition / Concept
3. Technical Explanation
4. Relationship Understanding

Rules:
- Every answer must be supported ONLY by the supplied evidence.
- Do not require outside knowledge.
- Do not invent values, dates, components or relationships.
- Questions must be useful for a real engineering knowledge assistant.
- Avoid trivial wording copied directly from the evidence.
- Reference answers must be concise but complete.
- Return ONLY valid JSON array.

Evidence:
{evidence}

Return:
[
  {{
    "question": "...",
    "reference_answer": "...",
    "question_type": "Single Fact"
  }},
  {{
    "question": "...",
    "reference_answer": "...",
    "question_type": "Definition / Concept"
  }},
  {{
    "question": "...",
    "reference_answer": "...",
    "question_type": "Technical Explanation"
  }},
  {{
    "question": "...",
    "reference_answer": "...",
    "question_type": "Relationship Understanding"
  }}
]
"""


def generate_benchmark():
    print("\nCreating fresh 100-question benchmark...")

    # Use balanced evidence across the 34 source documents.
    selected = []
    per_doc = max(1, math.ceil(TARGET_QUESTIONS / len(SOURCE_NAMES)))

    for filename in SOURCE_NAMES:
        candidates = [
            x for x in source_chunks
            if x["source_file"] == filename
        ]
        random.shuffle(candidates)
        selected.extend(candidates[:per_doc])

    random.shuffle(selected)

    # 20 batches x 4 = 80 questions.
    batches = selected[:20]

    generated = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []

        for item in batches:
            evidence = (
                f"SOURCE FILE: {item['source_file']}\n"
                f"CHUNK INDEX: {item['chunk_index']}\n"
                f"TEXT:\n{item['text']}"
            )

            futures.append(
                ex.submit(
                    llm_call,
                    QUESTION_GENERATION_PROMPT.format(evidence=evidence)
                )
            )

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()

            if not result["success"]:
                print(f"Benchmark batch failed: {result['error']}")
                continue

            try:
                text = result["text"].strip()
                if text.startswith("```"):
                    text = text.replace("```json", "").replace("```", "").strip()

                items = json.loads(text)

                if isinstance(items, list):
                    generated.extend(items)

            except Exception as e:
                print(f"Benchmark JSON parse failed: {e}")

            print(f"Benchmark batches: {i}/{len(futures)}")

    # Add multi-document, safety/regulatory, comparison and unanswerable questions.
    # These are generated from pairs of different source documents.
    pair_batches = []

    for _ in range(10):
        a, b = random.sample(source_chunks, 2)
        if a["source_file"] == b["source_file"]:
            continue

        pair_batches.append((a, b))

    SPECIAL_PROMPT = """
Create EXACTLY 2 benchmark questions from the two supplied source excerpts.

Question 1 type: Multi-Document Reasoning OR Comparison.
Question 2 type: Safety / Regulatory OR Unanswerable / Abstention.

For an unanswerable question, ask for a specific technical fact that is
NOT present in either excerpt and set the reference answer exactly to:
"Insufficient information in the source documents."

Return ONLY JSON array with:
question, reference_answer, question_type.

SOURCE A:
{a}

SOURCE B:
{b}
"""

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = [
            ex.submit(
                llm_call,
                SPECIAL_PROMPT.format(
                    a=a["text"],
                    b=b["text"]
                )
            )
            for a, b in pair_batches
        ]

        for future in as_completed(futures):
            result = future.result()
            if not result["success"]:
                continue

            try:
                text = result["text"].strip()
                if text.startswith("```"):
                    text = text.replace("```json", "").replace("```", "").strip()

                items = json.loads(text)
                if isinstance(items, list):
                    generated.extend(items)
            except Exception:
                pass

    # Normalize and enforce question-type quotas.
    cleaned = []
    seen = set()

    for item in generated:
        if not isinstance(item, dict):
            continue

        q = str(item.get("question", "")).strip()
        a = str(item.get("reference_answer", "")).strip()
        qt = str(item.get("question_type", "")).strip()

        if not q or not a or not qt:
            continue

        if q.lower() in seen:
            continue

        if qt not in QUESTION_TYPES:
            continue

        seen.add(q.lower())
        cleaned.append({
            "id": len(cleaned) + 1,
            "question": q,
            "reference_answer": a,
            "question_type": qt,
        })

    # If generation did not produce enough questions, generate direct
    # questions from individual chunks until 100 is reached.
    while len(cleaned) < TARGET_QUESTIONS:
        item = random.choice(source_chunks)

        prompt = f"""
Create ONE high-quality benchmark question from this SMR source text.

Choose one type from:
Single Fact, Definition / Concept, Technical Explanation,
Safety / Regulatory, Comparison.

The answer must be explicitly supported by the source.
Do not use outside knowledge.
Return ONLY JSON:
{{
  "question": "...",
  "reference_answer": "...",
  "question_type": "..."
}}

SOURCE:
{item["text"]}
"""

        result = llm_call(prompt)

        if not result["success"]:
            continue

        try:
            text = result["text"].strip()
            if text.startswith("```"):
                text = text.replace("```json", "").replace("```", "").strip()

            x = json.loads(text)
            q = str(x.get("question", "")).strip()

            if q and q.lower() not in seen and x.get("reference_answer"):
                cleaned.append({
                    "id": len(cleaned) + 1,
                    "question": q,
                    "reference_answer": str(x["reference_answer"]).strip(),
                    "question_type": x.get(
                        "question_type",
                        "Technical Explanation"
                    ),
                })
                seen.add(q.lower())

        except Exception:
            continue

    # Force exactly 100.
    cleaned = cleaned[:TARGET_QUESTIONS]

    # Add IDs again after trimming.
    for i, x in enumerate(cleaned, 1):
        x["id"] = i

    BENCHMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK_FILE.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"Created: {BENCHMARK_FILE}")
    print(f"Frozen benchmark size: {len(cleaned)}")

    return cleaned


if benchmark_is_valid(BENCHMARK_FILE):
    benchmark = json.loads(
        BENCHMARK_FILE.read_text(encoding="utf-8")
    )[:TARGET_QUESTIONS]
    print(f"Using existing frozen benchmark: {len(benchmark)} questions")
else:
    benchmark = generate_benchmark()


if len(benchmark) < TARGET_QUESTIONS:
    raise RuntimeError(
        f"Benchmark has only {len(benchmark)} valid questions. "
        f"Need {TARGET_QUESTIONS}."
    )


# ============================================================
# 7. LOAD VECTOR DATABASES
# ============================================================

rag_db = chromadb.PersistentClient(path=str(RAG_DB))
rag_collection = rag_db.get_collection(RAG_COLLECTION)

wiki_db = chromadb.PersistentClient(path=str(WIKI_DB))
wiki_collections = wiki_db.list_collections()

if not wiki_collections:
    raise RuntimeError("No Wiki collection found. Run build_wiki_index.py first.")

wiki_collection = max(
    wiki_collections,
    key=lambda c: c.count()
)

print(f"RAG vectors: {rag_collection.count()}")
print(f"Wiki vectors: {wiki_collection.name} ({wiki_collection.count()})")


# ============================================================
# 8. RETRIEVAL
# ============================================================

def embed(text):
    return embedding_model.encode(
        [text],
        normalize_embeddings=True
    )[0].tolist()


def retrieve_rag(question, where=None):
    kwargs = {
        "query_embeddings": [embed(question)],
        "n_results": RAG_TOP_K,
        "include": ["documents", "metadatas"]
    }

    if where:
        kwargs["where"] = where

    result = rag_collection.query(**kwargs)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    return [
        {"text": d, "metadata": m or {}}
        for d, m in zip(docs, metas)
    ]


def retrieve_wiki(question, where=None):
    kwargs = {
        "query_embeddings": [embed(question)],
        "n_results": WIKI_TOP_K,
        "include": ["documents", "metadatas"]
    }

    if where:
        kwargs["where"] = where

    result = wiki_collection.query(**kwargs)

    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    return [
        {"text": d, "metadata": m or {}}
        for d, m in zip(docs, metas)
    ]


# ============================================================
# 9. ANSWERING
# ============================================================

def build_answer_prompt(question, contexts, system_name):
    context_text = "\n\n".join(
        f"[SOURCE {i + 1}]\n{x['text']}"
        for i, x in enumerate(contexts)
    )

    return f"""
You are answering a question using a Rolls-Royce SMR engineering
knowledge system.

SYSTEM: {system_name}

QUESTION:
{question}

EVIDENCE:
{context_text}

Rules:
- Use ONLY the evidence supplied.
- Do not use external knowledge.
- Do not invent facts, values, dates or relationships.
- If the evidence is insufficient, answer exactly:
  "Insufficient information in the source documents."
- Be concise, technically precise and complete.

ANSWER:
"""


def answer_question(item, system_name, where=None):
    question = item["question"]

    start = time.perf_counter()

    if system_name == "RAG":
        contexts = retrieve_rag(question, where)
    else:
        contexts = retrieve_wiki(question, where)

    retrieval_latency = time.perf_counter() - start

    prompt = build_answer_prompt(
        question,
        contexts,
        system_name
    )

    result = llm_call(prompt)

    return {
        "id": item["id"],
        "question": question,
        "question_type": item["question_type"],
        "reference_answer": item["reference_answer"],
        "system": system_name,
        "answer": result["text"],
        "prompt_tokens": result["prompt_tokens"],
        "completion_tokens": result["completion_tokens"],
        "total_tokens": result["total_tokens"],
        "generation_latency": result["latency"],
        "retrieval_latency": retrieval_latency,
        "total_latency": (
            retrieval_latency + result["latency"]
            if result["latency"] is not None
            else None
        ),
        "success": result["success"],
        "error": result["error"],
        "contexts": contexts,
    }


# ============================================================
# 10. LLM-AS-JUDGE
# ============================================================

JUDGE_PROMPT = """
You are an independent evaluator for Rolls-Royce SMR technical
knowledge retrieval.

QUESTION:
{question}

REFERENCE ANSWER:
{reference}

SYSTEM ANSWER:
{answer}

Evaluate from 0 to 10:

- factual_accuracy
- semantic_similarity
- completeness
- relevance
- hallucination_resistance
- technical_correctness

Also provide:
- abstention_correct: 0 or 1
- overall_quality: 0 to 10

For an unanswerable question, the correct behaviour is to explicitly
state that the source documents do not contain enough information.

Return ONLY JSON:
{{
  "factual_accuracy": 0,
  "semantic_similarity": 0,
  "completeness": 0,
  "relevance": 0,
  "hallucination_resistance": 0,
  "technical_correctness": 0,
  "abstention_correct": 0,
  "overall_quality": 0,
  "reason": ""
}}
"""


def judge_answer(record):
    result = llm_call(
        JUDGE_PROMPT.format(
            question=record["question"],
            reference=record["reference_answer"],
            answer=record["answer"]
        )
    )

    if not result["success"]:
        return {
            **record,
            "judge_success": False,
            "judge_error": result["error"]
        }

    text = result["text"].strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    try:
        score = json.loads(text)

        return {
            **record,
            **score,
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
# 11. PHASE 1 — QUESTION-TYPE QUALITY
# ============================================================

def run_quality():
    print("\n" + "=" * 75)
    print("PHASE 1 — QUESTION TYPE / ANSWER QUALITY")
    print("=" * 75)

    raw = []
    judged = []

    for system in ["RAG", "WikiLLM"]:
        print(f"\nRunning {system} — {len(benchmark)} questions")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [
                ex.submit(answer_question, x, system)
                for x in benchmark
            ]

            for i, future in enumerate(as_completed(futures), 1):
                try:
                    record = future.result()
                    raw.append(record)

                    if record["success"]:
                        judged.append(judge_answer(record))

                except Exception as e:
                    print("Question failed:", e)

                print(f"{system}: {i}/{len(benchmark)}")

    out = RESULT_DIR / "quality_results.json"
    out.write_text(
        json.dumps(judged, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )

    return judged


# ============================================================
# 12. DOCUMENT SCALING
# ============================================================

def metadata_document_key(meta):
    if not isinstance(meta, dict):
        return None

    for key in [
        "source_file",
        "source",
        "file",
        "filename",
        "document",
        "document_name",
        "source_path"
    ]:
        if key in meta and meta[key]:
            return str(meta[key])

    return None


def get_collection_document_names(collection):
    try:
        sample = collection.get(
            limit=min(collection.count(), 1000),
            include=["metadatas"]
        )
    except Exception:
        return set()

    names = set()

    for meta in sample.get("metadatas", []):
        key = metadata_document_key(meta)

        if key:
            names.add(Path(key).name)

    return names


RAG_META_DOCS = get_collection_document_names(rag_collection)
WIKI_META_DOCS = get_collection_document_names(wiki_collection)

print(f"RAG metadata document mapping: {len(RAG_META_DOCS)}")
print(f"Wiki metadata document mapping: {len(WIKI_META_DOCS)}")


def make_where_for_docs(collection_docs, selected_docs):
    """
    Try to construct a Chroma metadata filter.
    If metadata does not expose document identity, return None.
    """

    if not collection_docs:
        return None

    available = set(collection_docs)
    selected = [
        x for x in selected_docs
        if x in available
    ]

    if not selected:
        return None

    # Try common metadata keys.
    # The actual query is tested by the caller.
    return {"source_file": {"$in": selected}}


def choose_document_subset(n):
    # Deterministic, representative sampling.
    rng = random.Random(RANDOM_SEED + n)

    files = SOURCE_NAMES.copy()
    rng.shuffle(files)

    return files[:n]


def run_document_scaling():
    print("\n" + "=" * 75)
    print("PHASE 2 — DOCUMENT SCALE")
    print("=" * 75)

    results = []

    for n_docs in DOCUMENT_LEVELS:
        selected_docs = choose_document_subset(n_docs)

        print(f"\nDocument level: {n_docs}/{len(SOURCE_NAMES)}")

        # Select questions whose source documents are likely represented
        # when source_file metadata exists. Otherwise use the full frozen
        # benchmark and clearly mark the scale as retrieval-corpus only.
        for system in ["RAG", "WikiLLM"]:

            valid_where = None

            try:
                if system == "RAG":
                    valid_where = make_where_for_docs(
                        RAG_META_DOCS,
                        selected_docs
                    )
                else:
                    valid_where = make_where_for_docs(
                        WIKI_META_DOCS,
                        selected_docs
                    )

                # Verify filter before using it.
                if valid_where:
                    test = (
                        retrieve_rag("SMR", valid_where)
                        if system == "RAG"
                        else retrieve_wiki("SMR", valid_where)
                    )

                    if not test:
                        valid_where = None

            except Exception:
                valid_where = None

            # Use a representative 20-question subset for each scale.
            sample_questions = benchmark[:20]

            rows = []

            for item in sample_questions:
                r = answer_question(
                    item,
                    system,
                    valid_where
                )
                rows.append(r)

            successful = [r for r in rows if r["success"]]

            results.append({
                "documents": n_docs,
                "system": system,
                "questions": len(rows),
                "successful": len(successful),
                "success_rate": (
                    100 * len(successful) / len(rows)
                    if rows else 0
                ),
                "avg_quality": None,
                "avg_latency": (
                    statistics.mean(
                        r["total_latency"]
                        for r in successful
                        if r["total_latency"] is not None
                    )
                    if any(
                        r["total_latency"] is not None
                        for r in successful
                    )
                    else None
                ),
                "avg_prompt_tokens": (
                    statistics.mean(
                        r["prompt_tokens"]
                        for r in successful
                    )
                    if successful else 0
                ),
                "avg_completion_tokens": (
                    statistics.mean(
                        r["completion_tokens"]
                        for r in successful
                    )
                    if successful else 0
                ),
                "avg_total_tokens": (
                    statistics.mean(
                        r["total_tokens"]
                        for r in successful
                    )
                    if successful else 0
                ),
                "filter_used": bool(valid_where),
            })

            print(
                f"{system}: {len(successful)}/{len(rows)} successful"
            )

    # Quality at document scale is evaluated from the same quality
    # benchmark where possible. This keeps judge calls controlled.
    quality = {}
    qpath = RESULT_DIR / "quality_results.json"

    if qpath.exists():
        qrows = json.loads(qpath.read_text(encoding="utf-8"))
        for system in ["RAG", "WikiLLM"]:
            quality[system] = statistics.mean(
                x["overall_quality"]
                for x in qrows
                if x.get("system") == system
                and x.get("judge_success")
            ) if any(
                x.get("system") == system
                and x.get("judge_success")
                for x in qrows
            ) else None

    for row in results:
        row["quality_score"] = quality.get(row["system"])

    out = RESULT_DIR / "document_scaling_results.json"
    out.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    return results


# ============================================================
# 13. PHASE 3 — QUERY FREQUENCY
# ============================================================

def run_frequency():
    print("\n" + "=" * 75)
    print("PHASE 3 — INCREASING QUERY FREQUENCY")
    print("=" * 75)

    results = []

    # No repeated LLM judging here.
    # This phase measures operational behaviour only.

    for multiplier in FREQUENCY_LEVELS:
        workload = benchmark * multiplier

        print(f"\nFrequency: {multiplier}x = {len(workload)} queries")

        for system in ["RAG", "WikiLLM"]:

            start = time.perf_counter()
            completed = []

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [
                    ex.submit(answer_question, x, system)
                    for x in workload
                ]

                for i, future in enumerate(as_completed(futures), 1):
                    try:
                        completed.append(future.result())
                    except Exception as e:
                        completed.append({
                            "success": False,
                            "error": str(e),
                            "total_latency": None,
                            "total_tokens": 0
                        })

                    if i % 10 == 0 or i == len(workload):
                        print(f"{system}: {i}/{len(workload)}")

            elapsed = time.perf_counter() - start

            successful = [
                x for x in completed
                if x.get("success")
            ]

            latencies = [
                x["total_latency"]
                for x in successful
                if x.get("total_latency") is not None
            ]

            tokens = [
                x.get("total_tokens", 0)
                for x in successful
            ]

            results.append({
                "frequency": multiplier,
                "system": system,
                "queries": len(workload),
                "successful": len(successful),
                "failed": len(workload) - len(successful),
                "success_rate": (
                    100 * len(successful) / len(workload)
                    if workload else 0
                ),
                "elapsed_seconds": elapsed,
                "throughput_qps": (
                    len(successful) / elapsed
                    if elapsed else 0
                ),
                "avg_latency": (
                    statistics.mean(latencies)
                    if latencies else None
                ),
                "p95_latency": (
                    float(np.percentile(latencies, 95))
                    if latencies else None
                ),
                "avg_total_tokens": (
                    statistics.mean(tokens)
                    if tokens else 0
                ),
                "total_tokens": sum(tokens),
            })

    out = RESULT_DIR / "frequency_results.json"
    out.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8"
    )

    return results


# ============================================================
# 14. STATISTICAL / VERDICT HELPERS
# ============================================================

def bootstrap_ci(values, seed=42, n=3000):
    values = [float(x) for x in values if x is not None]

    if len(values) < 2:
        return None, None

    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=float)

    means = [
        np.mean(
            rng.choice(arr, size=len(arr), replace=True)
        )
        for _ in range(n)
    ]

    return (
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5))
    )


def question_type_summary(quality_rows):
    output = []

    for qtype in QUESTION_TYPES:
        row = {
            "question_type": qtype
        }

        for system in ["RAG", "WikiLLM"]:
            vals = [
                x["overall_quality"]
                for x in quality_rows
                if x.get("question_type") == qtype
                and x.get("system") == system
                and x.get("judge_success")
            ]

            row[system] = (
                statistics.mean(vals)
                if vals else None
            )

        if row["RAG"] is not None and row["WikiLLM"] is not None:
            row["delta_RAG_minus_Wiki"] = (
                row["RAG"] - row["WikiLLM"]
            )

            row["winner"] = (
                "RAG"
                if row["RAG"] > row["WikiLLM"]
                else "WikiLLM"
                if row["WikiLLM"] > row["RAG"]
                else "Tie"
            )
        else:
            row["delta_RAG_minus_Wiki"] = None
            row["winner"] = "Insufficient data"

        output.append(row)

    return output


def overall_quality_summary(rows):
    result = {}

    for system in ["RAG", "WikiLLM"]:
        vals = [
            x["overall_quality"]
            for x in rows
            if x.get("system") == system
            and x.get("judge_success")
        ]

        ci = bootstrap_ci(vals)

        result[system] = {
            "quality": statistics.mean(vals) if vals else None,
            "ci_low": ci[0] if ci else None,
            "ci_high": ci[1] if ci else None,
            "n": len(vals),
        }

    return result


def create_final_verdict(quality_rows, scale_rows, freq_rows, qtypes):
    qs = overall_quality_summary(quality_rows)

    quality_rag = qs["RAG"]["quality"]
    quality_wiki = qs["WikiLLM"]["quality"]

    # Quality category wins
    rag_type_wins = sum(x["winner"] == "RAG" for x in qtypes)
    wiki_type_wins = sum(x["winner"] == "WikiLLM" for x in qtypes)

    # Frequency wins based on efficiency:
    # throughput 35%, reliability 30%, latency 20%, tokens 15%.
    freq_rows = [x for x in freq_rows]

    freq_wins = {"RAG": 0, "WikiLLM": 0}
    frequency_detail = []

    for f in FREQUENCY_LEVELS:
        a = next(x for x in freq_rows if x["frequency"] == f and x["system"] == "RAG")
        b = next(x for x in freq_rows if x["frequency"] == f and x["system"] == "WikiLLM")

        max_tp = max(a["throughput_qps"], b["throughput_qps"], 1e-9)
        max_lat = max(a["avg_latency"] or 1, b["avg_latency"] or 1)
        max_tok = max(a["avg_total_tokens"], b["avg_total_tokens"], 1)

        def score(x):
            tp = x["throughput_qps"] / max_tp
            rel = x["success_rate"] / 100
            lat = 1 - ((x["avg_latency"] or max_lat) / max_lat)
            tok = 1 - (x["avg_total_tokens"] / max_tok)
            return 0.35 * tp + 0.30 * rel + 0.20 * max(0, lat) + 0.15 * max(0, tok)

        sr = score(a)
        sw = score(b)

        winner = "RAG" if sr > sw else "WikiLLM" if sw > sr else "Tie"

        if winner != "Tie":
            freq_wins[winner] += 1

        frequency_detail.append({
            "frequency": f,
            "RAG_efficiency": sr,
            "WikiLLM_efficiency": sw,
            "winner": winner,
        })

    # Overall efficiency summary
    rag_eff = statistics.mean(x["RAG_efficiency"] for x in frequency_detail)
    wiki_eff = statistics.mean(x["WikiLLM_efficiency"] for x in frequency_detail)

    # Quality normalized to 0..1
    rag_q = quality_rag / 10
    wiki_q = quality_wiki / 10

    # Reliability
    rag_rel = statistics.mean(
        x["success_rate"] for x in freq_rows if x["system"] == "RAG"
    ) / 100

    wiki_rel = statistics.mean(
        x["success_rate"] for x in freq_rows if x["system"] == "WikiLLM"
    ) / 100

    # Architecture decision:
    # 50% answer quality, 30% operational efficiency, 20% reliability.
    rag_overall = 0.50 * rag_q + 0.30 * rag_eff + 0.20 * rag_rel
    wiki_overall = 0.50 * wiki_q + 0.30 * wiki_eff + 0.20 * wiki_rel

    # Hybrid is recommended when strengths are genuinely complementary:
    # one system wins answer quality and the other wins efficiency,
    # or question-type wins are substantially split.
    quality_winner = "RAG" if rag_q > wiki_q else "WikiLLM" if wiki_q > rag_q else "Tie"
    efficiency_winner = "RAG" if rag_eff > wiki_eff else "WikiLLM" if wiki_eff > rag_eff else "Tie"

    if quality_winner != efficiency_winner and quality_winner != "Tie":
        recommendation = "Hybrid"
        recommendation_reason = (
            f"{quality_winner} leads on answer quality while "
            f"{efficiency_winner} leads on operational efficiency. "
            "A hybrid architecture is therefore the strongest enterprise option."
        )
    else:
        recommendation = (
            "RAG" if rag_overall > wiki_overall
            else "WikiLLM" if wiki_overall > rag_overall
            else "Hybrid"
        )

        recommendation_reason = (
            f"{recommendation} has the strongest combined measured score "
            "for the evaluated SMR workload."
        )

    return {
        "RAG_quality": quality_rag,
        "WikiLLM_quality": quality_wiki,
        "RAG_quality_CI": f"{qs['RAG']['ci_low']:.2f}–{qs['RAG']['ci_high']:.2f}",
        "WikiLLM_quality_CI": f"{qs['WikiLLM']['ci_low']:.2f}–{qs['WikiLLM']['ci_high']:.2f}",
        "RAG_question_type_wins": rag_type_wins,
        "WikiLLM_question_type_wins": wiki_type_wins,
        "RAG_efficiency": rag_eff,
        "WikiLLM_efficiency": wiki_eff,
        "RAG_reliability": rag_rel,
        "WikiLLM_reliability": wiki_rel,
        "RAG_overall_score": rag_overall,
        "WikiLLM_overall_score": wiki_overall,
        "quality_winner": quality_winner,
        "efficiency_winner": efficiency_winner,
        "recommendation": recommendation,
        "recommendation_reason": recommendation_reason,
        "frequency_detail": frequency_detail,
    }


# ============================================================
# 15. PROFESSIONAL EXCEL REPORT
# ============================================================

NAVY = "17365D"
BLUE = "1F4E78"
LIGHT_BLUE = "D9EAF7"
GREY = "F2F2F2"
DARK_GREY = "595959"
GREEN = "E2F0D9"
YELLOW = "FFF2CC"
RED = "FCE4D6"
WHITE = "FFFFFF"


def style_header(ws, row, start_col, end_col):
    fill = PatternFill("solid", fgColor=BLUE)
    font = Font(color=WHITE, bold=True)
    for col in range(start_col, end_col + 1):
        cell = ws.cell(row=row, column=col)
        cell.fill = fill
        cell.font = font
        cell.alignment = Alignment(horizontal="center", vertical="center")


def autosize(ws, max_width=42):
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        width = max(
            len(str(c.value)) if c.value is not None else 0
            for c in col
        )
        ws.column_dimensions[letter].width = min(max(width + 2, 12), max_width)


def add_title(ws, title, subtitle=None):
    ws["A1"] = title
    ws["A1"].font = Font(
        size=20,
        bold=True,
        color=WHITE
    )
    ws["A1"].fill = PatternFill(
        "solid",
        fgColor=NAVY
    )
    ws.merge_cells("A1:H1")

    if subtitle:
        ws["A2"] = subtitle
        ws["A2"].font = Font(
            italic=True,
            color=DARK_GREY
        )
        ws.merge_cells("A2:H2")


def create_excel(
    quality_rows,
    scale_rows,
    freq_rows,
    qtypes,
    verdict
):
    print("\nCreating professional Excel report...")

    wb = Workbook()
    ws = wb.active
    ws.title = "Executive Summary"

    # --------------------------------------------------------
    # Executive Summary
    # --------------------------------------------------------

    add_title(
        ws,
        "RAG vs WikiLLM — Rolls-Royce SMR Evaluation",
        "Controlled benchmark: answer quality, scalability, query frequency, efficiency and architecture recommendation"
    )

    ws["A4"] = "FINAL RECOMMENDATION"
    ws["A4"].font = Font(size=14, bold=True, color=WHITE)
    ws["A4"].fill = PatternFill("solid", fgColor=BLUE)

    ws["B4"] = verdict["recommendation"]
    ws["B4"].font = Font(size=16, bold=True)
    ws["B4"].fill = PatternFill(
        "solid",
        fgColor=GREEN if verdict["recommendation"] == "RAG"
        else YELLOW if verdict["recommendation"] == "WikiLLM"
        else LIGHT_BLUE
    )

    ws.merge_cells("B4:D4")

    ws["A6"] = "Decision Area"
    ws["B6"] = "RAG"
    ws["C6"] = "WikiLLM"
    ws["D6"] = "Winner / Recommendation"
    style_header(ws, 6, 1, 4)

    decision_rows = [
        (
            "Overall Answer Quality",
            verdict["RAG_quality"],
            verdict["WikiLLM_quality"],
            "RAG" if verdict["RAG_quality"] > verdict["WikiLLM_quality"]
            else "WikiLLM"
            if verdict["WikiLLM_quality"] > verdict["RAG_quality"]
            else "Tie"
        ),
        (
            "Question-Type Coverage",
            verdict["RAG_question_type_wins"],
            verdict["WikiLLM_question_type_wins"],
            "RAG" if verdict["RAG_question_type_wins"] > verdict["WikiLLM_question_type_wins"]
            else "WikiLLM"
            if verdict["WikiLLM_question_type_wins"] > verdict["RAG_question_type_wins"]
            else "Balanced"
        ),
        (
            "Frequency Efficiency",
            verdict["RAG_efficiency"],
            verdict["WikiLLM_efficiency"],
            verdict["efficiency_winner"]
        ),
        (
            "Reliability",
            verdict["RAG_reliability"],
            verdict["WikiLLM_reliability"],
            "RAG" if verdict["RAG_reliability"] > verdict["WikiLLM_reliability"]
            else "WikiLLM"
        ),
        (
            "Combined Decision Score",
            verdict["RAG_overall_score"],
            verdict["WikiLLM_overall_score"],
            verdict["recommendation"]
        ),
    ]

    for r, row in enumerate(decision_rows, 7):
        for c, value in enumerate(row, 1):
            ws.cell(r, c).value = value

    ws["A14"] = "Recommendation rationale"
    ws["A14"].font = Font(bold=True)
    ws["B14"] = verdict["recommendation_reason"]
    ws["B14"].alignment = Alignment(wrap_text=True)
    ws.merge_cells("B14:H16")

    ws["A18"] = "What the company should use it for"
    ws["A18"].font = Font(bold=True, color=WHITE)
    ws["A18"].fill = PatternFill("solid", fgColor=BLUE)

    recommendation_matrix = [
        ("High-frequency factual lookup", "RAG"),
        ("Exact values / specific facts", "RAG"),
        ("Frequently changing documents", "RAG"),
        ("Concept discovery", "WikiLLM"),
        ("Cross-document reasoning", "WikiLLM"),
        ("Relationship understanding", "WikiLLM"),
        ("Enterprise knowledge exploration", "WikiLLM"),
        ("Mixed operational workload", "Hybrid"),
    ]

    for c, h in enumerate(["Scenario", "Recommended Architecture"], 1):
        ws.cell(19, c).value = h
    style_header(ws, 19, 1, 2)

    for r, row in enumerate(recommendation_matrix, 20):
        ws.cell(r, 1).value = row[0]
        ws.cell(r, 2).value = row[1]

    # --------------------------------------------------------
    # Question Type Analysis
    # --------------------------------------------------------

    qt = wb.create_sheet("Question Type Analysis")
    add_title(
        qt,
        "Question-Type Performance",
        "Higher score = better answer quality (0–10)"
    )

    headers = [
        "Question Type",
        "RAG Score",
        "WikiLLM Score",
        "Delta (RAG-Wiki)",
        "Winner"
    ]

    for c, h in enumerate(headers, 1):
        qt.cell(4, c).value = h
    style_header(qt, 4, 1, 5)

    for r, row in enumerate(qtypes, 5):
        qt.cell(r, 1).value = row["question_type"]
        qt.cell(r, 2).value = row["RAG"]
        qt.cell(r, 3).value = row["WikiLLM"]
        qt.cell(r, 4).value = row["delta_RAG_minus_Wiki"]
        qt.cell(r, 5).value = row["winner"]

    qt.conditional_formatting.add(
        f"B5:C{4 + len(qtypes)}",
        ColorScaleRule(
            start_type="min",
            start_color="F8696B",
            mid_type="percentile",
            mid_value=50,
            mid_color="FFEB84",
            end_type="max",
            end_color="63BE7B"
        )
    )

    chart = BarChart()
    chart.title = "Answer Quality by Question Type"
    chart.y_axis.title = "Score / 10"
    chart.x_axis.title = "Question Type"

    data = Reference(
        qt,
        min_col=2,
        max_col=3,
        min_row=4,
        max_row=4 + len(qtypes)
    )
    cats = Reference(
        qt,
        min_col=1,
        min_row=5,
        max_row=4 + len(qtypes)
    )

    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 9
    chart.width = 18
    qt.add_chart(chart, "G4")

    # --------------------------------------------------------
    # Document Scaling
    # --------------------------------------------------------

    ds = wb.create_sheet("Document Scaling")
    add_title(
        ds,
        "Performance as Document Volume Increases",
        "Tests 5, 10, 20 and 34 source documents; token, latency and reliability trends"
    )

    headers = [
        "Documents",
        "System",
        "Questions",
        "Success Rate %",
        "Quality Score",
        "Avg Latency (s)",
        "Avg Prompt Tokens",
        "Avg Completion Tokens",
        "Avg Total Tokens",
        "Filter Used"
    ]

    for c, h in enumerate(headers, 1):
        ds.cell(4, c).value = h
    style_header(ds, 4, 1, len(headers))

    for r, row in enumerate(scale_rows, 5):
        values = [
            row["documents"],
            row["system"],
            row["questions"],
            row["success_rate"],
            row["quality_score"],
            row["avg_latency"],
            row["avg_prompt_tokens"],
            row["avg_completion_tokens"],
            row["avg_total_tokens"],
            "Yes" if row["filter_used"] else "No",
        ]

        for c, value in enumerate(values, 1):
            ds.cell(r, c).value = value

    chart = LineChart()
    chart.title = "Average Tokens vs Document Volume"
    chart.y_axis.title = "Tokens / Query"
    chart.x_axis.title = "Documents"

    rag_rows = [
        r for r in range(5, 5 + len(scale_rows))
        if ds.cell(r, 2).value == "RAG"
    ]

    # Build chart from the complete table using rows for RAG/Wiki.
    # Excel will display the two system series.
    data = Reference(
        ds,
        min_col=9,
        min_row=4,
        max_row=4 + len(scale_rows)
    )
    cats = Reference(
        ds,
        min_col=1,
        min_row=5,
        max_row=4 + len(scale_rows)
    )
    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 8
    chart.width = 17
    ds.add_chart(chart, "L4")

    # --------------------------------------------------------
    # Frequency
    # --------------------------------------------------------

    fq = wb.create_sheet("Query Frequency")
    add_title(
        fq,
        "Performance as Query Frequency Increases",
        "1×, 2×, 4× and 8× repeated workload against the same frozen benchmark"
    )

    headers = [
        "Frequency",
        "System",
        "Queries",
        "Success Rate %",
        "Throughput QPS",
        "Avg Latency (s)",
        "P95 Latency (s)",
        "Avg Total Tokens",
        "Total Tokens"
    ]

    for c, h in enumerate(headers, 1):
        fq.cell(4, c).value = h
    style_header(fq, 4, 1, len(headers))

    for r, row in enumerate(freq_rows, 5):
        vals = [
            row["frequency"],
            row["system"],
            row["queries"],
            row["success_rate"],
            row["throughput_qps"],
            row["avg_latency"],
            row["p95_latency"],
            row["avg_total_tokens"],
            row["total_tokens"],
        ]
        for c, value in enumerate(vals, 1):
            fq.cell(r, c).value = value

    chart = LineChart()
    chart.title = "Average Latency as Query Frequency Increases"
    chart.y_axis.title = "Seconds"
    chart.x_axis.title = "Frequency Multiplier"

    data = Reference(
        fq,
        min_col=6,
        min_row=4,
        max_row=4 + len(freq_rows)
    )
    cats = Reference(
        fq,
        min_col=1,
        min_row=5,
        max_row=4 + len(freq_rows)
    )

    chart.add_data(data, titles_from_data=True)
    chart.set_categories(cats)
    chart.height = 8
    chart.width = 17
    fq.add_chart(chart, "K4")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    met = wb.create_sheet("Metrics")
    add_title(
        met,
        "Evaluation Metrics",
        "Metrics used to support the architecture decision"
    )

    metric_rows = [
        ("Factual Accuracy", "0–10", "Correctness of stated facts"),
        ("Semantic Similarity", "0–10", "Alignment with the reference meaning"),
        ("Completeness", "0–10", "Coverage of required information"),
        ("Relevance", "0–10", "Directness of the answer"),
        ("Hallucination Resistance", "0–10", "Avoidance of unsupported claims"),
        ("Technical Correctness", "0–10", "Technical validity"),
        ("Abstention Accuracy", "0–1", "Correct refusal when evidence is absent"),
        ("Average Latency", "seconds", "End-to-end response time"),
        ("P95 Latency", "seconds", "95th percentile response time"),
        ("Throughput", "queries/sec", "Completed queries per second"),
        ("Prompt Tokens", "tokens/query", "Input-token consumption"),
        ("Completion Tokens", "tokens/query", "Output-token consumption"),
        ("Total Tokens", "tokens/query", "Combined token consumption"),
        ("Success Rate", "%", "Requests completed without failure"),
        ("Question-Type Win Rate", "%", "Share of categories won"),
    ]

    for c, h in enumerate(["Metric", "Unit", "Purpose"], 1):
        met.cell(4, c).value = h
    style_header(met, 4, 1, 3)

    for r, row in enumerate(metric_rows, 5):
        for c, value in enumerate(row, 1):
            met.cell(r, c).value = value

    # --------------------------------------------------------
    # Raw Quality
    # --------------------------------------------------------

    raw = wb.create_sheet("Raw Quality Results")
    add_title(
        raw,
        "Per-Question Quality Results",
        "Underlying judged results used to produce the summary"
    )

    raw_headers = [
        "ID", "System", "Question Type", "Question",
        "Factual", "Semantic", "Completeness", "Relevance",
        "Hallucination", "Technical", "Abstention", "Overall Quality"
    ]

    for c, h in enumerate(raw_headers, 1):
        raw.cell(4, c).value = h
    style_header(raw, 4, 1, len(raw_headers))

    for r, x in enumerate(quality_rows, 5):
        vals = [
            x.get("id"),
            x.get("system"),
            x.get("question_type"),
            x.get("question"),
            x.get("factual_accuracy"),
            x.get("semantic_similarity"),
            x.get("completeness"),
            x.get("relevance"),
            x.get("hallucination_resistance"),
            x.get("technical_correctness"),
            x.get("abstention_correct"),
            x.get("overall_quality"),
        ]

        for c, value in enumerate(vals, 1):
            raw.cell(r, c).value = value

    raw.freeze_panes = "A5"
    raw.auto_filter.ref = f"A4:L{4 + len(quality_rows)}"

    # --------------------------------------------------------
    # Final Architecture Decision
    # --------------------------------------------------------

    dec = wb.create_sheet("Architecture Decision")
    add_title(
        dec,
        "Architecture Decision Matrix",
        "Use this sheet for the final Rolls-Royce engineering/management discussion"
    )

    rows = [
        ("High-frequency factual lookup", "RAG", "Low latency, bounded retrieval and stable token use"),
        ("Exact values / specific facts", "RAG", "Direct source-level retrieval"),
        ("Frequently changing documents", "RAG", "Simpler incremental updates"),
        ("Concept discovery", "WikiLLM", "Concept-centric organization"),
        ("Cross-document reasoning", "WikiLLM", "Broader conceptual relationships"),
        ("Relationship understanding", "WikiLLM", "Concept-to-concept navigation"),
        ("Large, stable knowledge exploration", "WikiLLM", "Useful when conceptual coverage outweighs retrieval overhead"),
        ("Mixed enterprise workload", "Hybrid", "Route factual/high-frequency queries to RAG and complex exploratory queries to WikiLLM"),
    ]

    for c, h in enumerate(["Scenario", "Recommended", "Rationale"], 1):
        dec.cell(4, c).value = h
    style_header(dec, 4, 1, 3)

    for r, row in enumerate(rows, 5):
        for c, value in enumerate(row, 1):
            dec.cell(r, c).value = value

    dec["A15"] = "Measured recommendation"
    dec["A15"].font = Font(bold=True, color=WHITE)
    dec["A15"].fill = PatternFill("solid", fgColor=BLUE)

    dec["B15"] = verdict["recommendation"]
    dec["B15"].font = Font(size=15, bold=True)

    dec["A17"] = "Why"
    dec["A17"].font = Font(bold=True)

    dec["B17"] = verdict["recommendation_reason"]
    dec["B17"].alignment = Alignment(wrap_text=True)
    dec.merge_cells("B17:H20")

    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    for sheet in wb.worksheets:
        sheet.freeze_panes = sheet.freeze_panes or "A4"

        for row in sheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(
                    vertical="top",
                    wrap_text=True
                )

        autosize(sheet)

        # Light borders for used cells.
        thin = Side(style="thin", color="D9E1F2")
        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    cell.border = Border(
                        left=thin,
                        right=thin,
                        top=thin,
                        bottom=thin
                    )

    # Number formats
    for sheet_name in ["Question Type Analysis", "Document Scaling", "Query Frequency"]:
        wsx = wb[sheet_name]
        for row in wsx.iter_rows():
            for cell in row:
                if isinstance(cell.value, float):
                    cell.number_format = "0.00"

    wb.save(EXCEL_FILE)
    print(f"Excel report created: {EXCEL_FILE}")


# ============================================================
# 16. MAIN
# ============================================================

def main():
    print("\n" + "=" * 75)
    print("ROLLS-ROYCE SMR — RAG vs WikiLLM CONTROLLED EVALUATION")
    print("=" * 75)

    # Phase 1: answer quality by question type.
    quality_rows = run_quality()

    # Phase 2: document volume.
    scale_rows = run_document_scaling()

    # Phase 3: increasing query frequency.
    freq_rows = run_frequency()

    # Question-type summary.
    qtypes = question_type_summary(quality_rows)

    # Final measured architecture recommendation.
    verdict = create_final_verdict(
        quality_rows,
        scale_rows,
        freq_rows,
        qtypes
    )

    # JSON summary.
    final_json = {
        "benchmark_questions": len(benchmark),
        "document_levels": DOCUMENT_LEVELS,
        "frequency_levels": FREQUENCY_LEVELS,
        "question_type_summary": qtypes,
        "verdict": verdict,
    }

    (RESULT_DIR / "final_verdict.json").write_text(
        json.dumps(
            final_json,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # Professional Excel.
    create_excel(
        quality_rows,
        scale_rows,
        freq_rows,
        qtypes,
        verdict
    )

    print("\n" + "=" * 75)
    print("FINAL MEASURED RESULT")
    print("=" * 75)
    print(f"Quality winner: {verdict['quality_winner']}")
    print(f"Efficiency winner: {verdict['efficiency_winner']}")
    print(f"Recommendation: {verdict['recommendation']}")
    print(verdict["recommendation_reason"])
    print("=" * 75)

    print("\nGenerated:")
    print(f"  {BENCHMARK_FILE}")
    print(f"  {RESULT_DIR / 'quality_results.json'}")
    print(f"  {RESULT_DIR / 'document_scaling_results.json'}")
    print(f"  {RESULT_DIR / 'frequency_results.json'}")
    print(f"  {RESULT_DIR / 'final_verdict.json'}")
    print(f"  {EXCEL_FILE}")


if __name__ == "__main__":
    main()
