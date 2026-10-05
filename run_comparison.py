import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chromadb
from openai import AzureOpenAI, RateLimitError
from sentence_transformers import SentenceTransformer

BENCHMARK = Path("evaluation/benchmark.json")
OUTPUT = Path("evaluation/raw_results.json")
RAG_CHROMA_DIR = Path("./chroma_store")
RAG_COLLECTION = "extracted-doc"
WIKI_CHROMA_DIR = Path("./wiki_chroma_store")
WIKI_COLLECTION = "wikillm-index"
WIKI_PAGES = Path("wiki/pages")
EMBEDDING_MODEL = "all-mpnet-base-v2"

AZURE_ENDPOINT = "https://ease-azure-ai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_API_KEY = "PASTE_YOUR_API_KEY_HERE"

RAG_TOP_K = 30
WIKI_INDEX_TOP_K = 30
WIKI_TOP_CONCEPTS = 5
WIKI_MAX_RELATED = 10
MAX_PAGE_CHARS = 4500
MAX_WIKI_CONTEXT_CHARS = 50000
WORKERS = 4

RAG_PROMPT = """You are answering questions about Rolls-Royce SMR engineering and regulatory documentation.
Use ONLY the supplied context. Do not use outside knowledge.
If the answer is not supported by the context, say exactly: "Insufficient information in the retrieved documents."
Be precise, technically grounded, and concise.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:"""

WIKI_PROMPT = """You are answering questions using a concept-centric Rolls-Royce SMR knowledge wiki.
Use ONLY the supplied concept pages. Combine information across related concepts when supported.
Do not use outside knowledge.
If the answer is not present, say exactly: "Insufficient information in retrieved concepts."
Be precise, technically grounded, and concise.

QUESTION:
{question}

CONCEPT CONTEXT:
{context}

ANSWER:"""


def make_client():
    if AZURE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise SystemExit("Put your Azure API key in AZURE_API_KEY first.")
    return AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )


def generate(client, prompt, retries=4):
    last = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=1200,
            )
            usage = r.usage
            return (
                r.choices[0].message.content or "",
                int(getattr(usage, "prompt_tokens", 0) or 0),
                int(getattr(usage, "completion_tokens", 0) or 0),
            )
        except RateLimitError as e:
            last = e
            time.sleep(min(5 * (attempt + 1), 30))
        except Exception as e:
            last = e
            time.sleep(2)
    raise last


def page_content(slug):
    path = WIKI_PAGES / f"{slug}.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def related_slugs(content, known):
    if not content:
        return []
    found = re.findall(r"\[\[([^\]]+)\]\]", content)
    out = []
    seen = set()
    for slug in found:
        slug = slug.strip()
        if slug in known and slug not in seen:
            seen.add(slug)
            out.append(slug)
    return out


def wiki_context(top_slugs, known):
    selected = []
    visited = set()
    parts = []

    def add(slug):
        if slug in visited:
            return
        content = page_content(slug)
        if content is None:
            return
        visited.add(slug)
        selected.append(slug)
        clipped = content[:MAX_PAGE_CHARS]
        parts.append(f"===== CONCEPT: {slug} =====\n{clipped}")

    for slug in top_slugs:
        add(slug)
        content = page_content(slug)
        for rel in related_slugs(content, known)[:WIKI_MAX_RELATED]:
            add(rel)

    context = "\n\n".join(parts)
    if len(context) > MAX_WIKI_CONTEXT_CHARS:
        context = context[:MAX_WIKI_CONTEXT_CHARS]
    return context, selected


def main():
    if not BENCHMARK.exists():
        raise SystemExit("evaluation/benchmark.json not found. Run generate_eval_dataset.py first.")
    if OUTPUT.exists():
        raise SystemExit("evaluation/raw_results.json already exists. Delete it only to intentionally rerun.")

    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))
    records = benchmark["records"]

    print("Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    rag_client = chromadb.PersistentClient(path=str(RAG_CHROMA_DIR))
    rag = rag_client.get_collection(name=RAG_COLLECTION)

    wiki_client = chromadb.PersistentClient(path=str(WIKI_CHROMA_DIR))
    wiki = wiki_client.get_collection(name=WIKI_COLLECTION)
    known_slugs = {m.get("slug") for m in wiki.get(include=["metadatas"])["metadatas"] if m and m.get("slug")}

    print(f"RAG vectors : {rag.count()}")
    print(f"Wiki pages  : {wiki.count()}")
    print(f"Questions   : {len(records)}")
    print(f"Workers     : {WORKERS}")

    def run_one(record):
        client = make_client()
        question = record["question"]
        qemb = model.encode([question], normalize_embeddings=True)[0].tolist()

        # ---------------- RAG ----------------
        rr = rag.query(
            query_embeddings=[qemb],
            n_results=RAG_TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        rag_docs = rr["documents"][0]
        rag_meta = rr["metadatas"][0]
        rag_context = "\n\n".join(
            f"===== SOURCE CHUNK {i+1} =====\n{doc}"
            for i, doc in enumerate(rag_docs)
        )
        rag_prompt = RAG_PROMPT.format(question=question, context=rag_context)
        rag_answer, rag_prompt_tokens, rag_completion_tokens = generate(client, rag_prompt)

        # ---------------- WikiLLM-style ----------------
        wr = wiki.query(
            query_embeddings=[qemb],
            n_results=WIKI_INDEX_TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        wiki_meta = wr["metadatas"][0]
        top_slugs = []
        for m in wiki_meta:
            slug = m.get("slug") if m else None
            if slug and slug not in top_slugs:
                top_slugs.append(slug)
            if len(top_slugs) == WIKI_TOP_CONCEPTS:
                break

        wiki_context_text, selected_slugs = wiki_context(top_slugs, known_slugs)
        wiki_prompt = WIKI_PROMPT.format(question=question, context=wiki_context_text)
        wiki_answer, wiki_prompt_tokens, wiki_completion_tokens = generate(client, wiki_prompt)

        return {
            **record,
            "rag": {
                "answer": rag_answer,
                "prompt_tokens": rag_prompt_tokens,
                "completion_tokens": rag_completion_tokens,
                "total_tokens": rag_prompt_tokens + rag_completion_tokens,
                "retrieved": [
                    {
                        "source_file": m.get("source_file") if m else None,
                        "chunk_index": m.get("chunk_index") if m else None,
                        "distance": rr["distances"][0][i],
                    }
                    for i, m in enumerate(rag_meta)
                ],
            },
            "wikillm": {
                "answer": wiki_answer,
                "prompt_tokens": wiki_prompt_tokens,
                "completion_tokens": wiki_completion_tokens,
                "total_tokens": wiki_prompt_tokens + wiki_completion_tokens,
                "top_concepts": top_slugs,
                "expanded_concepts": selected_slugs,
            },
        }

    results = [None] * len(records)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_one, r): i for i, r in enumerate(records)}
        for n, fut in enumerate(as_completed(futures), 1):
            idx = futures[fut]
            results[idx] = fut.result()
            print(f"Completed {n}/{len(records)}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"records": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDONE: {OUTPUT}")


if __name__ == "__main__":
    main()
