import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import AzureOpenAI, RateLimitError

INPUT_DIR = Path(r"C:\Users\Praveena\merlin-rag\extracted_doc")
OUTPUT_DIR = Path("evaluation")
OUTPUT_FILE = OUTPUT_DIR / "benchmark.json"

AZURE_ENDPOINT = "https://ease-azure-ai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_API_KEY = "PASTE_YOUR_API_KEY_HERE"

CHUNK_SIZE = 3500
CHUNK_OVERLAP = 200
TOTAL_QUESTIONS = 50
ANSWERABLE_QUESTIONS = 45
UNANSWERABLE_QUESTIONS = 5
WORKERS = 6
SEED = 42

TYPES = [
    "exact_fact", "definition", "technical_explanation", "safety_regulatory",
    "multi_chunk_reasoning", "comparison", "relationship"
]


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}" if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > chunk_size:
                start = 0
                while start < len(para):
                    chunks.append(para[start:start + chunk_size])
                    start += chunk_size - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)
    out = []
    for i, c in enumerate(chunks):
        if i == 0:
            out.append(c)
        else:
            tail = chunks[i - 1][-overlap:] if overlap else ""
            out.append((tail + "\n\n" + c) if tail else c)
    return out


def load_chunks():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Input folder not found: {INPUT_DIR}")
    files = sorted(INPUT_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {INPUT_DIR}")
    all_chunks = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        for i, chunk in enumerate(chunk_text(text)):
            all_chunks.append({
                "source_file": f.name,
                "chunk_index": i,
                "text": chunk,
            })
    return all_chunks


def make_client():
    if AZURE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise SystemExit("Put your Azure API key in AZURE_API_KEY first.")
    return AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )


def call_json(client, prompt, retries=4):
    last = None
    for attempt in range(retries):
        try:
            try:
                r = client.chat.completions.create(
                    model=AZURE_DEPLOYMENT,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_completion_tokens=900,
                )
            except Exception as e:
                if "response_format" in str(e).lower() or "unsupported" in str(e).lower() or "400" in str(e):
                    r = client.chat.completions.create(
                        model=AZURE_DEPLOYMENT,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=900,
                    )
                else:
                    raise
            raw = r.choices[0].message.content or "{}"
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    return json.loads(m.group(0), strict=False)
                raise
        except RateLimitError as e:
            last = e
            time.sleep(min(5 * (attempt + 1), 30))
        except Exception as e:
            last = e
            time.sleep(2)
    raise last


def build_prompt(item, qtype, negative=False):
    if negative:
        task = """
Create ONE plausible technical question whose answer is definitely NOT stated in the source text.
The question must ask for a specific fact, value, date, limit, component, or relationship that the
source does not provide. Do not invent an answer. The reference answer must be exactly:
"Insufficient information in the source documents."
"""
    else:
        task = f"""
Create ONE high-quality benchmark question of type: {qtype}.
The answer MUST be supported by the source text. Prefer questions that require understanding rather
than copying a sentence. Do not require knowledge outside the supplied source.
Write a concise reference answer using only the source text.
"""
    return f"""
You are creating a fixed benchmark for comparing Traditional RAG against a concept-centric WikiLLM
system for Rolls-Royce SMR engineering and regulatory documents.

{task}

SOURCE FILE: {item['source_file']}
CHUNK INDEX: {item['chunk_index']}

SOURCE TEXT:
"""
{item['text']}
"""

Return ONLY JSON in this exact shape:
{{
  "question": "...",
  "reference_answer": "...",
  "question_type": "{('unanswerable' if negative else qtype)}"
}}
"""


def main():
    if OUTPUT_FILE.exists():
        raise SystemExit(
            f"{OUTPUT_FILE} already exists. It is the frozen benchmark. Delete it manually only if you intentionally want a new benchmark."
        )

    chunks = load_chunks()
    rng = random.Random(SEED)

    # Prefer broad document coverage: sample roughly evenly from the 34 files.
    by_file = {}
    for item in chunks:
        by_file.setdefault(item["source_file"], []).append(item)
    files = sorted(by_file)
    selected = []
    for f in files:
        selected.append(rng.choice(by_file[f]))
    remaining = [x for x in chunks if x not in selected]
    rng.shuffle(remaining)
    selected = selected[:ANSWERABLE_QUESTIONS] + remaining[:max(0, ANSWERABLE_QUESTIONS - len(selected))]
    selected = selected[:ANSWERABLE_QUESTIONS]

    jobs = []
    for i, item in enumerate(selected):
        jobs.append((i, item, TYPES[i % len(TYPES)], False))
    for i in range(UNANSWERABLE_QUESTIONS):
        item = rng.choice(chunks)
        jobs.append((ANSWERABLE_QUESTIONS + i, item, "unanswerable", True))

    print(f"Total source chunks available: {len(chunks)}")
    print(f"Generating {len(jobs)} benchmark questions with {WORKERS} workers...")

    results = [None] * len(jobs)

    def worker(job):
        idx, item, qtype, negative = job
        client = make_client()
        prompt = build_prompt(item, qtype, negative)
        data = call_json(client, prompt)
        question = str(data.get("question", "")).strip()
        answer = str(data.get("reference_answer", "")).strip()
        if not question or not answer:
            raise ValueError("LLM returned an empty question/reference answer")
        if negative:
            answer = "Insufficient information in the source documents."
        return idx, {
            "id": f"q{idx + 1:03d}",
            "question": question,
            "reference_answer": answer,
            "question_type": "unanswerable" if negative else str(data.get("question_type", qtype)),
            "source_file": item["source_file"],
            "source_chunk": item["chunk_index"],
        }

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(worker, job) for job in jobs]
        for n, future in enumerate(as_completed(futures), 1):
            idx, record = future.result()
            results[idx] = record
            print(f"Generated {n}/{len(jobs)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "benchmark_size": len(results),
        "answerable": ANSWERABLE_QUESTIONS,
        "unanswerable": UNANSWERABLE_QUESTIONS,
        "source_documents": len(files),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "seed": SEED,
        "records": results,
    }
    OUTPUT_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDONE: {OUTPUT_FILE}")
    print("FREEZE THIS FILE. Use the same benchmark for both systems.")


if __name__ == "__main__":
    main()
