"""
build_wiki_from_file.py

Builds a concept-centric markdown wiki (Rolls-Royce SMR domain) from a SINGLE
already-extracted text file, following the WikiLLM pattern
(https://github.com/ssurya1696/WikiLLM):

    text file  -->  chunk  -->  embed (Ollama: nomic-embed-text)
               -->  store in a FRESH local Chroma collection (scoped to this file only)
               -->  for each chunk, call Ollama (mistral) with the concept-extraction
                    prompt -> parse JSON -> write/merge one markdown page per concept
               -->  build an index.md summarising every page (>= 2 line description each)

Requirements (Ollama must already be running locally):
    ollama pull mistral
    ollama pull nomic-embed-text

Usage (from an activated venv):
    python build_wiki_from_file.py --input path\\to\\your_extracted_text.txt

All other paths default to subfolders of the current working directory
(intended to be run from C:\\Users\\Praveena\\merlin-rag):
    wiki\\            <- generated markdown wiki pages + index.md
    chroma_store\\    <- fresh, file-scoped Chroma persistent DB
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
# 1. Read + chunk the source text file
# --------------------------------------------------------------------------

def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


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

def ollama_embed(text: str, model: str = DEFAULT_EMBED_MODEL):
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def ollama_generate(prompt: str, model: str = DEFAULT_LLM_MODEL) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_ctx": 8192},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["response"]


# --------------------------------------------------------------------------
# 3. The customised extraction prompt (verbatim, as supplied)
# --------------------------------------------------------------------------

def extract_concepts_for_chunk(chunk_text_value, llm_model=None):
    prompt = f"""

You are maintaining a persistent Rolls-Royce Small Modular Reactor, SMR, engineering and regulatory wiki.

This is a concept-centric nuclear engineering knowledge system.

A single source document may update MANY concept pages.

Your task is to extract ALL meaningful technical, regulatory, safety, design, engineering, organisational, document, and process concepts from the supplied source text and create/update one markdown wiki page per concept.

IMPORTANT:
- Return ONLY valid JSON.
- Each concept/entity must become its own page.
- Prefer over-extraction rather than under-extraction.
- Use the exact terminology from the source text wherever possible.
- Preserve acronyms exactly as written, e.g. SMR, E3S, GDA, ALARP, BAT, SSC.
- If the source expands an acronym, include the expansion in the page.
- Do not invent facts not supported by the source text.
- If a concept is only partially defined in the source, create a cautious page using only available context.
- Use YAML frontmatter.
- Use lowercase slugs for wiki links.
- Use [[slug]] backlinks.
- Create backlinks aggressively between related concepts.
- Preserve traceability to the source text.
- Keep knowledge cumulative and append-only.
- Prefer granular pages over broad merged pages.
- If unsure whether something deserves a page, CREATE THE PAGE.
- Do not output narrative prose outside the JSON object.

ENTITY EXTRACTION REQUIREMENTS

You MUST extract ALL possible entities and concepts including, but not limited to:

- Rolls-Royce SMR
- SMR design concepts
- E3S concepts
- environment concepts
- safety concepts
- security concepts
- safeguards concepts
- GDA concepts
- regulatory assessment concepts
- nuclear safety case concepts
- design safety claims
- safety functions
- safety measures
- safety classifications
- design basis events
- design basis faults
- beyond design basis events
- severe accidents
- hazards
- internal hazards
- external hazards
- hazard assessments
- risk assessments
- ALARP arguments
- BAT arguments
- deterministic safety analysis
- probabilistic safety analysis
- fault studies
- transient analysis
- accident analysis
- radiological consequences
- source terms
- dose limits
- dose targets
- radiation protection concepts
- radioactive waste concepts
- spent fuel concepts
- fuel design concepts
- reactor core concepts
- reactivity control concepts
- shutdown systems
- protection systems
- control systems
- instrumentation and control systems
- reactor coolant systems
- emergency cooling systems
- containment systems
- ventilation systems
- electrical systems
- civil structures
- mechanical systems
- auxiliary systems
- support systems
- safety systems
- safety-related systems
- structures, systems, and components, SSCs
- components
- equipment
- modules
- assemblies
- materials
- manufacturing processes
- construction processes
- commissioning processes
- operating processes
- maintenance processes
- inspection processes
- testing processes
- surveillance requirements
- operational limits and conditions
- design requirements
- regulatory requirements
- acceptance criteria
- standards
- codes
- guidance documents
- submissions
- reports
- topic reports
- chapters
- claims
- arguments
- evidence
- assumptions
- constraints
- interfaces
- dependencies
- engineering changes
- design options
- optioneering decisions
- design maturity concepts
- verification activities
- validation activities
- quality assurance concepts
- management arrangements
- organisational entities
- regulators
- requesting parties
- assessment bodies
- suppliers
- site-related assumptions
- generic site envelope concepts
- security arrangements
- safeguards arrangements
- environmental permits
- discharges
- emissions
- effluents
- monitoring arrangements
- mitigations
- corrective actions
- open points
- regulatory observations
- regulatory issues
- assessment findings

EVERY meaningful concept should become a page if:
- it may be referenced later
- it affects or constrains another concept
- it participates in relationships
- it is part of an engineering, safety, environmental, security, safeguards, or regulatory argument
- it may appear in future documents
- it is a named system, component, process, requirement, document, organisation, hazard, claim, standard, or assessment topic

RELATIONSHIP EXTRACTION

Extract relationships wherever supported by the source text.

Useful relationship types include:

- related_to
- part_of
- contains
- supports
- depends_on
- constrains
- affects
- mitigates
- protected_by
- controlled_by
- monitored_by
- assessed_by
- regulated_by
- required_by
- satisfies
- demonstrates
- claims
- evidenced_by
- references
- implements
- interfaces_with
- derived_from
- applies_to
- used_by
- operated_by
- maintained_by
- inspected_by
- tested_by
- verified_by
- validated_by
- supersedes
- superseded_by
- open_issue_for
- resolves
- contributes_to
- prevents
- reduces
- causes
- initiated_by
- results_in

PAGE FORMAT

Each generated page must use the following markdown format inside the JSON "content" field:

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
Concise wiki-style summary of the concept in 1-3 sentences.

## Definition
Clear definition grounded only in the supplied source text.

## Technical Details
- Important technical details from the source.
- Relevant design, safety, regulatory, environmental, operational, or engineering information.
- Include exact values, limits, document names, codes, or requirements where present.

## Role in Rolls-Royce SMR
Explain how this concept relates to the Rolls-Royce SMR, if supported by the source text.

## Safety, Environmental, Security, or Safeguards Relevance
Explain any E3S, safety, environmental, security, safeguards, risk, or regulatory significance.

## Relationships

### Related
- [[related-page]]

### Part Of
- [[parent-page]]

### Contains
- [[child-page]]

### Depends On
- [[dependency-page]]

### Interfaces With
- [[interface-page]]

### Regulated By
- [[regulator-or-requirement-page]]

### References
- [[document-or-standard-page]]

## Open Points
- Any uncertainty, incomplete information, open issue, assumption, or follow-up action stated or implied by the source.

## Sources
- Source text chunk

CATEGORY RULES

Use one of the following categories where possible:

- Reactor
- System
- Subsystem
- Component
- Structure
- Equipment
- Material
- Fuel
- Process
- Procedure
- Requirement
- Safety
- Security
- Safeguards
- Environment
- Regulation
- Standard
- Document
- Organisation
- Regulator
- Assessment
- Hazard
- Fault
- Accident
- Risk
- Claim
- Evidence
- Interface
- Assumption
- Constraint
- Operation
- Maintenance
- Inspection
- Test
- Waste
- RadiologicalProtection
- QualityAssurance
- Programme
- Other

IMPORTANCE RULES

Set importance as:

- Critical: central to nuclear safety, E3S, regulatory approval, reactor design, major safety function, major system, or licensing/GDA outcome.
- Important: materially relevant to design, operation, engineering substantiation, assessment, or compliance.
- Reference: supporting detail, background context, minor term, or low-level referenced item.

SLUG RULES

- Slugs must be lowercase.
- Replace spaces and punctuation with hyphens.
- Remove duplicate hyphens.
- Preserve meaningful acronyms in lowercase.
- Examples:
  - "Generic Design Assessment" -> "generic-design-assessment"
  - "E3S Case" -> "e3s-case"
  - "Reactor Coolant System" -> "reactor-coolant-system"

CONTENT RULES

- Use exact source terminology wherever possible.
- Do not merge separate systems, requirements, documents, or concepts into one page.
- Do not invent dates, statuses, values, or relationships.
- If a field is unknown, use an empty list or state "Not specified in the source text."
- Use markdown tables where useful for parameters, requirements, limits, or document references.
- Create links using [[slug-name]].
- Backlink aggressively to related pages.
- Include source traceability in the YAML frontmatter and Sources section.
- Output must be machine-parseable JSON.

JSON OUTPUT FORMAT

Return exactly this JSON structure:

{{
  "pages": [
    {{
      "title": "Page Title",
      "slug": "page-slug",
      "category": "System",
      "importance": "Critical",
      "aliases": ["Alias 1", "Alias 2"],
      "relationships": {{
        "related_to": ["related-page-slug"],
        "part_of": ["parent-page-slug"],
        "contains": ["child-page-slug"],
        "depends_on": ["dependency-page-slug"],
        "interfaces_with": ["interface-page-slug"],
        "regulated_by": ["regulator-or-requirement-slug"],
        "references": ["document-or-standard-slug"]
      }},
      "content": "FULL MARKDOWN PAGE"
    }}
  ]
}}

If no meaningful concepts are present, return:

{{
  "pages": []
}}

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
# 5. Chroma storage (fresh, file-scoped)
# --------------------------------------------------------------------------

class OllamaEmbeddingFunction:
    """Chroma-compatible embedding function backed by a local Ollama model."""

    def __init__(self, model: str = DEFAULT_EMBED_MODEL):
        self.model = model

    def __call__(self, input):
        return [ollama_embed(text, self.model) for text in input]

    def name(self):
        return f"ollama-{self.model}"


def get_chroma_collection(persist_dir: str, collection_name: str, embed_model: str):
    client = chromadb.PersistentClient(path=persist_dir)
    ef = OllamaEmbeddingFunction(model=embed_model)
    # fresh collection scoped to this single file: wipe any previous run
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name, embedding_function=ef)


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


def write_or_merge_page(wiki_dir: Path, page: dict, chunk_index: int, source_label: str):
    slug = page.get("slug") or slugify(page.get("title", f"untitled-{chunk_index}"))
    slug = slugify(slug)
    out_path = wiki_dir / f"{slug}.md"
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

def extract_summary(md_text: str) -> str:
    m = re.search(r"##\s*Summary\s*\n(.*?)(?=\n##|\Z)", md_text, re.DOTALL)
    if m:
        summary = " ".join(m.group(1).strip().split())
        return summary
    return "No summary extracted."


def build_index(wiki_dir: Path):
    pages = []
    for path in sorted(wiki_dir.glob("*.md")):
        if path.name == "index.md":
            continue
        text = path.read_text(encoding="utf-8")
        fields, _ = split_frontmatter(text)
        title = fields.get("title", path.stem)
        category = fields.get("category", "Other")
        importance = fields.get("importance", "Reference")
        summary = extract_summary(text)
        n_sources = len(fields.get("sources", []))
        pages.append({
            "slug": path.stem, "title": title, "category": category,
            "importance": importance, "summary": summary, "n_sources": n_sources,
        })

    pages.sort(key=lambda p: (p["category"], p["title"]))

    lines = ["# Rolls-Royce SMR Wiki — Index", "",
             f"Total concept pages: {len(pages)}", ""]
    current_category = None
    for p in pages:
        if p["category"] != current_category:
            current_category = p["category"]
            lines.append(f"\n## {current_category}\n")
        lines.append(f"### [{p['title']}]({p['slug']}.md)")
        lines.append(f"{p['summary']}")
        lines.append(f"*Importance: {p['importance']} | Extracted from {p['n_sources']} source chunk(s).*")
        lines.append("")
    (wiki_dir / "index.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# 8. Orchestration
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a concept wiki from a single extracted text file.")
    parser.add_argument("--input", required=True, help="Path to the extracted .txt source file")
    parser.add_argument("--wiki-dir", default="wiki", help="Output folder for wiki pages (default: ./wiki)")
    parser.add_argument("--chroma-dir", default="chroma_store", help="Chroma persist dir (default: ./chroma_store)")
    parser.add_argument("--collection", default=None, help="Chroma collection name (default: derived from filename)")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL, help="Ollama model for extraction (default: mistral)")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--skip-chroma", action="store_true", help="Skip embedding/storage, only build the wiki")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    wiki_dir = Path(args.wiki_dir)
    wiki_dir.mkdir(parents=True, exist_ok=True)

    collection_name = args.collection or slugify(input_path.stem)

    print(f"[1/4] Reading {input_path} ...")
    text = read_text_file(input_path)

    print("[2/4] Chunking text ...")
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)
    print(f"      -> {len(chunks)} chunks")

    if not args.skip_chroma:
        print(f"[3/4] Embedding chunks into fresh Chroma collection '{collection_name}' "
              f"(model: {args.embed_model}) ...")
        collection = get_chroma_collection(args.chroma_dir, collection_name, args.embed_model)
        ids, docs, metadatas = [], [], []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{collection_name}-chunk-{i:04d}"
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append({"source_file": str(input_path.name), "chunk_index": i})
        collection.add(ids=ids, documents=docs, metadatas=metadatas)
        print(f"      -> stored {len(chunks)} chunks in {args.chroma_dir}/")
    else:
        print("[3/4] Skipping Chroma embedding/storage (--skip-chroma)")

    print(f"[4/4] Extracting concepts with Ollama model '{args.llm_model}' and writing wiki pages ...")
    total_pages_written = 0
    for i, chunk in enumerate(chunks):
        source_label = f"{input_path.name}#chunk-{i:04d}"
        prompt = extract_concepts_for_chunk(chunk, args.llm_model)

        parsed = None
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            raw = ollama_generate(prompt, model=args.llm_model)
            parsed = parse_json_safely(raw)
            if parsed is not None and "pages" in parsed:
                break
            print(f"      chunk {i}: JSON parse failed (attempt {attempt}/{MAX_LLM_RETRIES}), retrying ...")
            time.sleep(1)

        if parsed is None or "pages" not in parsed:
            print(f"      chunk {i}: giving up after {MAX_LLM_RETRIES} attempts, skipping.")
            continue

        pages = parsed.get("pages", [])
        for page in pages:
            try:
                slug, title, category, importance = write_or_merge_page(wiki_dir, page, i, source_label)
                total_pages_written += 1
                print(f"      chunk {i}: wrote/merged '{title}' ({category}/{importance}) -> {slug}.md")
            except Exception as e:
                print(f"      chunk {i}: failed to write page ({page.get('title')}): {e}")

    print("Building index.md ...")
    build_index(wiki_dir)

    print(f"\nDone. {len(list(wiki_dir.glob('*.md'))) - 1} unique wiki pages in '{wiki_dir}/' "
          f"({total_pages_written} page-writes total across {len(chunks)} chunks).")
    print(f"Index: {wiki_dir / 'index.md'}")


if __name__ == "__main__":
    main()
