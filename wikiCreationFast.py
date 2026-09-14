"""
build_wiki_from_folder.py

Same pipeline as build_wiki_from_file.py, but processes every .txt file in a
FOLDER instead of a single file, writing into one shared wiki (all concepts
merge together across every source document, same as if they were all one
big file):

    folder of .txt files  -->  chunk each file  -->  embed (Ollama: nomic-embed-text)
                           -->  store in a Chroma collection (scoped to the whole folder)
                           -->  for each chunk (across all files), call Ollama (mistral)
                                with the concept-extraction prompt -> parse JSON
                                -> write/merge one markdown page per concept
                           -->  build INDEX.md summarising every page (>= 2 line description each)
                           -->  append every ingestion to log.md

Requirements (Ollama must already be running locally):
    ollama pull mistral
    ollama pull nomic-embed-text

Usage (from an activated venv):
    python build_wiki_from_folder.py --input-dir path\\to\\your_text_files

All other paths default to subfolders of the current working directory:
    wiki\\pages\\      <- generated markdown concept pages
    wiki\\INDEX.md     <- wikilinked index, regenerated every run
    wiki\\log.md       <- append-only ingestion log
    chroma_store\\     <- Chroma persistent DB (skip with --skip-chroma)
"""

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

import requests

try:
    import chromadb
except ImportError:
    print("chromadb is not installed. Run: pip install chromadb", file=sys.stderr)
    raise

OLLAMA_URL = "http://localhost:11434"
DEFAULT_LLM_MODEL = "mistral"
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 200
MAX_LLM_RETRIES = 3


# --------------------------------------------------------------------------
# 0. Terminal progress bar (no external deps — pure stdlib)
# --------------------------------------------------------------------------

class ProgressBar:
    """
    Single-line terminal progress bar with elapsed/ETA, safe to use
    alongside interleaved log lines via .write().
    """

    def __init__(self, total: int, desc: str = "", width: int = 30):
        self.total = max(total, 0)
        self.desc = desc
        self.width = width
        self.count = 0
        self.suffix = ""
        self.start = time.time()
        self._last_render_len = 0

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = max(int(seconds), 0)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def set_suffix(self, text: str):
        self.suffix = text

    def update(self, n: int = 1):
        self.count += n
        self._render()

    def _render(self):
        if self.total <= 0:
            return
        frac = min(self.count / self.total, 1.0)
        filled = int(self.width * frac)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        rate = self.count / elapsed if elapsed > 0 else 0
        remaining = (self.total - self.count) / rate if rate > 0 else 0
        line = (
            f"\r{self.desc} [{bar}] {self.count}/{self.total} "
            f"({frac * 100:5.1f}%)  elapsed {self._fmt_time(elapsed)}  "
            f"eta {self._fmt_time(remaining)}"
        )
        if self.suffix:
            line += f"  {self.suffix}"
        pad = max(self._last_render_len - len(line), 0)
        sys.stdout.write(line + (" " * pad))
        sys.stdout.flush()
        self._last_render_len = len(line)

    def write(self, text: str):
        """Print a log line above the bar without corrupting it."""
        sys.stdout.write("\r" + " " * self._last_render_len + "\r")
        print(text)
        self._render()

    def close(self):
        self._render()
        sys.stdout.write("\n")
        sys.stdout.flush()


# --------------------------------------------------------------------------
# 1. Read + chunk the source text file
# --------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def find_text_files(input_dir: Path):
    """All .txt files in the folder, sorted for a stable, repeatable chunk order."""
    return sorted(input_dir.glob("*.txt"))


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
               overlap: int = DEFAULT_CHUNK_OVERLAP):
    """Simple paragraph-aware sliding-window chunker (no external deps)."""
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
                # paragraph itself too big: hard-split it
                start = 0
                while start < len(para):
                    chunks.append(para[start:start + chunk_size])
                    start += chunk_size - overlap
                current = ""
            else:
                current = para
    if current:
        chunks.append(current)

    # apply overlap between consecutive chunks for extraction continuity
    overlapped = []
    for i, c in enumerate(chunks):
        if i == 0:
            overlapped.append(c)
        else:
            tail = chunks[i - 1][-overlap:] if overlap else ""
            overlapped.append((tail + "\n\n" + c) if tail else c)
    return overlapped


# --------------------------------------------------------------------------
# 2. Ollama helpers (embeddings + generation)
# --------------------------------------------------------------------------

def ollama_embed(text: str, model: str = DEFAULT_EMBED_MODEL, timeout: int = 120,
                  max_retries: int = 3):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["embedding"]
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt < max_retries:
                time.sleep(2)
    raise last_exc


def ollama_generate(prompt: str, model: str = DEFAULT_LLM_MODEL, timeout: int = 1800,
                     num_ctx: int = 16384) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": num_ctx},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]


# --------------------------------------------------------------------------
# 3. The customised extraction prompt (verbatim, as supplied)
# --------------------------------------------------------------------------

def extract_concepts_for_chunk(chunk_text_value, llm_model=None):
    prompt = f"""
You are maintaining a persistent Rolls-Royce Small Modular Reactor (SMR) engineering and regulatory wiki. This is a concept-centric knowledge system: each source chunk may update multiple concept pages.

Extract the meaningful technical, regulatory, safety, design, engineering, organisational, or process concepts from the supplied source text below and create one markdown wiki page per concept — covering things like: systems, components, safety functions/classifications, hazards, design basis events, requirements, standards, documents, organisations, regulators, assessments, claims, and processes.

RULES:
- Return ONLY valid JSON, no prose outside it.
- Extract at most the 6 MOST SIGNIFICANT concepts in this chunk (favor breadth across future chunks over exhaustiveness in one call).
- Use exact source terminology. Preserve acronyms as written (SMR, E3S, GDA, ALARP, BAT, SSC, etc.) and expand them if the source does.
- Do not invent facts not in the source text. If partially defined, write a cautious page using only available context.
- Slugs: lowercase, hyphenated, no duplicate hyphens (e.g. "Reactor Coolant System" -> "reactor-coolant-system").
- Use [[slug]] backlinks between related concepts where supported by the text.
- category: one of Reactor, System, Subsystem, Component, Structure, Equipment, Material, Fuel, Process, Procedure, Requirement, Safety, Security, Safeguards, Environment, Regulation, Standard, Document, Organisation, Regulator, Assessment, Hazard, Fault, Accident, Risk, Claim, Evidence, Interface, Assumption, Constraint, Operation, Maintenance, Inspection, Test, Waste, RadiologicalProtection, QualityAssurance, Programme, Other.
- importance: Critical (central to safety/E3S/licensing/major system), Important (materially relevant to design/ops/compliance), or Reference (background/minor term).

Each page's "content" field must be this exact markdown structure:

---
title: Example Title
slug: example-title
category: System
importance: Important
aliases: [Alternative Name, Acronym]
systems: [related-system-slug]
components: [related-component-slug]
related: [related-page-slug]
sources: [source-text-chunk]
status: Draft
tags: [smr, safety, gda]
---

# Example Title

## Summary
1-3 sentence wiki-style summary.

## Definition
Definition grounded only in the source text.

## Technical Details
- Key technical details, values, limits, document names, requirements from the source.

## Role in Rolls-Royce SMR
How this relates to the Rolls-Royce SMR, if supported by the source.

## Safety, Environmental, Security, or Safeguards Relevance
Any E3S/safety/security/safeguards/regulatory significance.

## Relationships

### Related
- [[related-page]]

### Part Of
- [[parent-page]]

### Depends On
- [[dependency-page]]

### Regulated By
- [[regulator-or-requirement-page]]

## Open Points
- Any uncertainty, open issue, or assumption implied by the source.

## Sources
- Source text chunk

Return exactly this JSON structure:

{{
  "pages": [
    {{
      "title": "Page Title",
      "slug": "page-slug",
      "category": "System",
      "importance": "Critical",
      "aliases": ["Alias 1"],
      "relationships": {{
        "related_to": ["related-page-slug"],
        "part_of": ["parent-page-slug"],
        "depends_on": ["dependency-page-slug"],
        "regulated_by": ["regulator-or-requirement-slug"]
      }},
      "content": "FULL MARKDOWN PAGE"
    }}
  ]
}}

If no meaningful concepts are present, return: {{"pages": []}}

SOURCE TEXT:
\"\"\"
{chunk_text_value}
\"\"\"
"""
    return prompt


# --------------------------------------------------------------------------
# 4. JSON parsing robustness
# --------------------------------------------------------------------------

def parse_json_safely(raw: str):
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw.strip())
    raw = re.sub(r"```$", "", raw.strip())
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # fallback: grab the outermost {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None


def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


# --------------------------------------------------------------------------
# 5. Chroma storage (resumable, batch-checkpointed)
# --------------------------------------------------------------------------

def load_key_set(path: Path) -> set:
    if path.exists():
        try:
            return set(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()


def save_key_set(path: Path, keys: set):
    path.write_text(json.dumps(sorted(keys)), encoding="utf-8")


def embed_and_store_chunks(all_chunks, chroma_dir, collection_name, embed_model,
                            batch_size=50, resume=False, time_budget_min=0):
    """
    Embeds chunks in small batches and commits each batch to Chroma
    immediately, checkpointing progress to disk after every batch.

    Safe to interrupt (Ctrl+C, crash, laptop shutdown): at most one
    in-flight batch is lost. Rerun with resume=True to continue exactly
    where it left off, without re-embedding anything already stored.
    """
    chroma_dir_path = Path(chroma_dir)
    chroma_dir_path.mkdir(parents=True, exist_ok=True)
    progress_path = chroma_dir_path / f"{collection_name}.embed_progress.json"

    client = chromadb.PersistentClient(path=str(chroma_dir_path))

    if resume:
        try:
            collection = client.get_collection(collection_name)
        except Exception:
            collection = client.create_collection(collection_name)
        done_keys = load_key_set(progress_path)
    else:
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
        collection = client.create_collection(collection_name)
        done_keys = set()
        if progress_path.exists():
            progress_path.unlink()

    remaining = [c for c in all_chunks if c["key"] not in done_keys]
    print(f"      {len(done_keys)}/{len(all_chunks)} chunk(s) already embedded "
          f"({len(remaining)} remaining).")

    run_start = time.time()
    stopped_early = False
    total_batches = (len(remaining) + batch_size - 1) // batch_size if remaining else 0

    pb = ProgressBar(total=len(remaining), desc="Embedding")

    for batch_num, i in enumerate(range(0, len(remaining), batch_size), start=1):
        if time_budget_min > 0:
            elapsed_min = (time.time() - run_start) / 60
            if elapsed_min >= time_budget_min:
                pb.write(f"Embedding time budget of {time_budget_min} min reached "
                         f"({len(done_keys)}/{len(all_chunks)} chunks embedded) — stopping. "
                         f"Rerun with --resume to continue.")
                stopped_early = True
                break

        batch = remaining[i:i + batch_size]
        ids = [f"{collection_name}-{c['key']}" for c in batch]
        docs = [c["text"] for c in batch]
        metadatas = [{"source_file": c["file_name"], "chunk_index": c["chunk_index"]} for c in batch]

        try:
            embeddings = [ollama_embed(doc, embed_model) for doc in docs]
        except requests.exceptions.RequestException as e:
            pb.write(f"Embedding batch {batch_num} failed ({e}) — stopping. "
                     f"Progress up to the last completed batch is saved. Rerun with --resume.")
            stopped_early = True
            break

        # upsert (not add): safe even if a previous crashed run partially
        # wrote this exact batch before dying.
        collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

        for c in batch:
            done_keys.add(c["key"])
        save_key_set(progress_path, done_keys)

        pb.set_suffix(f"batch {batch_num}/{total_batches}")
        pb.update(len(batch))

    pb.close()
    return collection, done_keys, stopped_early


# --------------------------------------------------------------------------
# 6. Wiki page writing / merging
# --------------------------------------------------------------------------

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
LIST_FIELD_RE = re.compile(r"^(\w+):\s*\[(.*)\]\s*$")


def split_frontmatter(md_text: str):
    m = FRONTMATTER_RE.match(md_text.strip() + "\n")
    if not m:
        return {}, md_text
    fm_raw, body = m.group(1), m.group(2)
    fields = {}
    for line in fm_raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        list_m = LIST_FIELD_RE.match(line)
        if list_m:
            key, items = list_m.groups()
            fields[key] = [i.strip().strip('"').strip("'") for i in items.split(",") if i.strip()]
        else:
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()
    return fields, body.strip()


def render_frontmatter(fields: dict) -> str:
    lines = ["---"]
    order = ["title", "slug", "category", "importance", "aliases", "systems",
             "components", "related", "sources", "status", "tags"]
    seen = set()
    for key in order:
        if key in fields:
            seen.add(key)
            val = fields[key]
            if isinstance(val, list):
                lines.append(f"{key}: [{', '.join(val)}]")
            else:
                lines.append(f"{key}: {val}")
    for key, val in fields.items():
        if key in seen:
            continue
        if isinstance(val, list):
            lines.append(f"{key}: [{', '.join(val)}]")
        else:
            lines.append(f"{key}: {val}")
    lines.append("---")
    return "\n".join(lines)


def merge_list_field(old_list, new_list):
    merged = list(old_list) if old_list else []
    for item in new_list or []:
        if item not in merged:
            merged.append(item)
    return merged


def write_or_merge_page(pages_dir: Path, page: dict, chunk_index: int, source_label: str):
    slug = page.get("slug") or slugify(page.get("title", f"untitled-{chunk_index}"))
    slug = slugify(slug)
    out_path = pages_dir / f"{slug}.md"
    new_content = page.get("content", "").strip()
    new_fields, new_body = split_frontmatter(new_content)
    # make sure sources include this chunk for traceability
    chunk_source = new_fields.get("sources") or [source_label]
    if source_label not in chunk_source:
        chunk_source = chunk_source + [source_label]
    new_fields["sources"] = chunk_source

    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        old_fields, old_body = split_frontmatter(existing)
        merged_fields = dict(old_fields)
        for key in ["aliases", "systems", "components", "related", "sources", "tags"]:
            merged_fields[key] = merge_list_field(old_fields.get(key, []), new_fields.get(key, []))
        # keep the strongest importance seen (Critical > Important > Reference)
        rank = {"Critical": 3, "Important": 2, "Reference": 1}
        old_imp, new_imp = old_fields.get("importance", "Reference"), new_fields.get("importance", "Reference")
        merged_fields["importance"] = old_imp if rank.get(old_imp, 0) >= rank.get(new_imp, 0) else new_imp
        for key in ["title", "slug", "category", "status"]:
            merged_fields[key] = old_fields.get(key) or new_fields.get(key)

        # strip the leading "# Title" H1 from the new body to avoid duplicate headers
        new_body_no_h1 = re.sub(r"^#\s+.*\n+", "", new_body, count=1)
        combined_body = (
            f"{old_body}\n\n---\n\n"
            f"## Additional Extraction ({source_label})\n\n"
            f"{new_body_no_h1}"
        )
        full_text = render_frontmatter(merged_fields) + "\n\n" + combined_body.strip() + "\n"
    else:
        full_text = render_frontmatter(new_fields) + "\n\n" + new_body.strip() + "\n"

    out_path.write_text(full_text, encoding="utf-8")
    return slug, new_fields.get("title", page.get("title", slug)), new_fields.get("category", "Other"), \
        new_fields.get("importance", "Reference")


# --------------------------------------------------------------------------
# 7. Index generation (>= 2 line description per page)
# --------------------------------------------------------------------------

def extract_summary_lines(md_text: str) -> list:
    """Return at least 2 lines of description for the index, per WikiLLM's format."""
    m = re.search(r"##\s*Summary\s*\n(.*?)(?=\n##|\Z)", md_text, re.DOTALL)
    if not m:
        return ["No summary extracted.", "See the full page for details."]
    raw = " ".join(m.group(1).strip().split())
    if not raw:
        return ["No summary extracted.", "See the full page for details."]
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw) if s.strip()]
    if len(sentences) >= 2:
        return sentences[:2]
    return [sentences[0], "See the full page for further detail and sources."]


def load_progress(wiki_dir: Path) -> set:
    progress_path = wiki_dir / ".progress.json"
    return load_key_set(progress_path)


def save_progress(wiki_dir: Path, done_keys: set):
    progress_path = wiki_dir / ".progress.json"
    save_key_set(progress_path, done_keys)


def build_index(wiki_dir: Path, pages_dir: Path):
    pages = []
    for path in sorted(pages_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        fields, _ = split_frontmatter(text)
        title = fields.get("title", path.stem)
        summary_lines = extract_summary_lines(text)
        pages.append({"slug": path.stem, "title": title, "summary_lines": summary_lines})

    pages.sort(key=lambda p: p["title"])

    lines = ["# Rolls-Royce SMR Wiki", "", "## Concept Pages", ""]
    for p in pages:
        lines.append(f"* [[{p['slug']}]] — {p['title']}")
        for sline in p["summary_lines"]:
            lines.append(f"  {sline}")
        lines.append("")
    (wiki_dir / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def append_log(wiki_dir: Path, chunk_index: int, source_label: str, written: list):
    """Append one entry per chunk to log.md, matching WikiLLM's ingestion log."""
    log_path = wiki_dir / "log.md"
    if not log_path.exists():
        log_path.write_text("# Wiki Ingestion Log\n\n", encoding="utf-8")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"## {timestamp} — {source_label}", ""]
    if written:
        for slug, title, category, importance in written:
            lines.append(f"- [{title}](pages/{slug}.md) ({category}/{importance})")
    else:
        lines.append("- (no pages written)")
    lines.append("")
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# 8. Orchestration
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a concept wiki + Chroma embeddings from every .txt file in a folder.")
    parser.add_argument("--input-dir", required=True, help="Folder containing extracted .txt source files")
    parser.add_argument("--wiki-dir", default="wiki", help="Output folder for wiki pages (default: ./wiki)")
    parser.add_argument("--chroma-dir", default="chroma_store", help="Chroma persist dir (default: ./chroma_store)")
    parser.add_argument("--collection", default=None, help="Chroma collection name (default: derived from folder name)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Ollama model for extraction (default: mistral)")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--skip-chroma", action="store_true", help="Skip embedding/storage, only build the wiki")
    parser.add_argument("--embed-batch-size", type=int, default=50,
                         help="Chunks embedded and committed to Chroma per batch (default: 50). "
                              "Progress is checkpointed after every batch.")
    parser.add_argument("--embed-time-budget-min", type=int, default=0,
                         help="Stop embedding after this many minutes (0 = no limit, run to completion). "
                              "Already-embedded chunks are kept in Chroma; rerun with --resume to continue. Default: 0.")
    parser.add_argument("--llm-timeout", type=int, default=1800,
                         help="Per-call timeout in seconds for the Ollama /api/generate call (default: 1800 = 30 min)")
    parser.add_argument("--num-ctx", type=int, default=16384,
                         help="Context window (tokens) passed to Ollama for generation (default: 16384). "
                              "Raise this if you increase --chunk-size, so the model has enough room left "
                              "for a large multi-page JSON response without truncating it.")
    parser.add_argument("--resume", action="store_true",
                         help="Resume a previous run: skip chunks already embedded (chroma) and/or "
                              "already recorded in <wiki-dir>/.progress.json (wiki generation).")
    parser.add_argument("--time-budget-min", type=int, default=20,
                         help="Stop starting new wiki-generation chunks after this many minutes total (0 = no limit). "
                              "Already-written pages are kept; rerun with --resume to continue later. Default: 20.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    text_files = find_text_files(input_dir)
    if not text_files:
        print(f"No .txt files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    wiki_dir = Path(args.wiki_dir)
    pages_dir = wiki_dir / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)

    collection_name = args.collection or slugify(input_dir.name)

    print(f"[1/4] Found {len(text_files)} .txt file(s) in {input_dir}")

    # Build one combined, ordered chunk list across every file. Each entry's
    # key ("filename#chunk-0004") is unique across the whole folder, so
    # progress/--resume tracks position across files, not just within one.
    print("[2/4] Chunking every file ...")
    all_chunks = []  # list of dicts: {key, file_name, chunk_index, text}
    for f in text_files:
        text = read_text_file(f)
        file_chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
        for i, c in enumerate(file_chunks):
            all_chunks.append({
                "key": f"{f.name}#chunk-{i:04d}",
                "file_name": f.name,
                "chunk_index": i,
                "text": c,
            })
        print(f"      {f.name}: {len(file_chunks)} chunks")
    print(f"      -> {len(all_chunks)} chunks total across {len(text_files)} file(s)")

    if not args.skip_chroma:
        print(f"[3/4] Embedding chunks into Chroma collection '{collection_name}' "
              f"(model: {args.embed_model}, batch size: {args.embed_batch_size}) ...")
        collection, embed_done_keys, embed_stopped_early = embed_and_store_chunks(
            all_chunks, args.chroma_dir, collection_name, args.embed_model,
            batch_size=args.embed_batch_size, resume=args.resume,
            time_budget_min=args.embed_time_budget_min,
        )
        print(f"      -> {len(embed_done_keys)}/{len(all_chunks)} chunks embedded in {args.chroma_dir}/")
        if embed_stopped_early:
            print("      NOTE: this collection has no embedding_function attached. When you query it "
                  "later for RAG, compute the query embedding yourself with ollama_embed(query_text) "
                  "and call collection.query(query_embeddings=[...]).")
    else:
        print("[3/4] Skipping Chroma embedding/storage (--skip-chroma)")

    print(f"[4/4] Extracting concepts with Ollama model '{args.llm_model}' and writing wiki pages ...")

    done_keys = load_progress(wiki_dir) if args.resume else set()
    if done_keys:
        print(f"      --resume: {len(done_keys)}/{len(all_chunks)} chunks already done, skipping those.")

    total_pages_written = 0
    run_start = time.time()
    stopped_early = False

    remaining_chunks = [c for c in all_chunks if c["key"] not in done_keys]
    pb = ProgressBar(total=len(remaining_chunks), desc="Wiki gen ")

    for c in remaining_chunks:
        key = c["key"]

        if args.time_budget_min > 0:
            elapsed_min = (time.time() - run_start) / 60
            if elapsed_min >= args.time_budget_min:
                pb.write(f"Time budget of {args.time_budget_min} min reached "
                         f"({len(done_keys)}/{len(all_chunks)} chunks done) — stopping early. "
                         f"Rerun with --resume to continue from here.")
                stopped_early = True
                break

        source_label = key  # e.g. "reportA.txt#chunk-0004"
        pb.set_suffix(f"current: {key}")
        prompt = extract_concepts_for_chunk(c["text"], args.llm_model)

        parsed = None
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            t0 = time.time()
            try:
                raw = ollama_generate(prompt, model=args.llm_model, timeout=args.llm_timeout,
                                       num_ctx=args.num_ctx)
            except requests.exceptions.Timeout:
                elapsed = time.time() - t0
                pb.write(f"{key}: Ollama call timed out after {elapsed:.0f}s "
                         f"(attempt {attempt}/{MAX_LLM_RETRIES}, limit {args.llm_timeout}s), retrying ...")
                time.sleep(2)
                continue
            except requests.exceptions.RequestException as e:
                pb.write(f"{key}: Ollama connection error ({e}) "
                         f"(attempt {attempt}/{MAX_LLM_RETRIES}), retrying ...")
                time.sleep(5)
                continue

            parsed = parse_json_safely(raw)
            if parsed is not None and "pages" in parsed:
                break
            pb.write(f"{key}: JSON parse failed (attempt {attempt}/{MAX_LLM_RETRIES}), retrying ...")
            time.sleep(1)

        if parsed is None or "pages" not in parsed:
            pb.write(f"{key}: giving up after {MAX_LLM_RETRIES} attempts, skipping "
                     f"(rerun with --resume once fixed, this chunk will be retried again).")
            pb.update(1)
            continue

        pages = parsed.get("pages", [])
        written = []
        for page in pages:
            try:
                slug, title, category, importance = write_or_merge_page(pages_dir, page, c["chunk_index"], source_label)
                total_pages_written += 1
                written.append((slug, title, category, importance))
                pb.write(f"{key}: wrote/merged '{title}' ({category}/{importance}) -> pages/{slug}.md")
            except Exception as e:
                pb.write(f"{key}: failed to write page ({page.get('title')}): {e}")

        append_log(wiki_dir, c["chunk_index"], source_label, written)
        done_keys.add(key)
        save_progress(wiki_dir, done_keys)
        pb.update(1)

    pb.close()
    print("Building INDEX.md ...")
    build_index(wiki_dir, pages_dir)

    print(f"\nDone. {len(list(pages_dir.glob('*.md')))} unique wiki pages in '{pages_dir}/' "
          f"({total_pages_written} page-writes total, {len(done_keys)}/{len(all_chunks)} chunks processed "
          f"across {len(text_files)} file(s)).")
    if stopped_early:
        print("Stopped early due to --time-budget-min. Rerun the same command with --resume to keep going.")
    print(f"Index: {wiki_dir / 'INDEX.md'}")
    print(f"Log: {wiki_dir / 'log.md'}")


if __name__ == "__main__":
    main()
