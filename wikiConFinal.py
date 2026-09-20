"""
wikiAzureFinal.py

Reads every .txt file in a folder, chunks it, and calls Azure OpenAI (your
original full extraction prompt) to build a concept wiki:

    folder of .txt files  -->  chunk each file  -->  for each chunk, call
    Azure OpenAI with the concept-extraction prompt -> parse JSON -> write/
    merge one markdown page per concept  -->  build INDEX.md (>= 2 line
    description per page)  -->  append every ingestion to log.md

No embedding/Chroma step here — this script is wiki generation only.
Embeddings (for the separate RAG part) are handled by a different script.

Usage (from an activated venv):
    python wikiAzureFinal.py

All defaults come from the config at the top of this file (input folder,
Azure endpoint/deployment/key). Output:
    wiki\\pages\\      <- generated markdown concept pages
    wiki\\INDEX.md     <- wikilinked index, regenerated every run
    wiki\\log.md       <- append-only ingestion log

Always auto-resumes: rerun the exact same command to continue after any
interruption. Use --fresh to ignore prior progress and start over.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from openai import AzureOpenAI, RateLimitError

# ============================================================
# AZURE OPENAI CONFIG — fill in your key below, everything
# else is already set from what you were given.
# ============================================================
AZURE_ENDPOINT = "https://ease-azure-ai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_MODEL_NAME = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_API_KEY = "PASTE_YOUR_API_KEY_HERE"   # <-- put your key between the quotes, nothing else

# Default folder to read source .txt files from if you don't pass --input-dir
DEFAULT_INPUT_DIR = r"C:\Users\Praveena\merlin-rag\extracted_docs"

DEFAULT_CHUNK_SIZE = 1800
DEFAULT_CHUNK_OVERLAP = 200
MAX_LLM_RETRIES = 3

_azure_client = None
_client_lock = threading.Lock()


def get_azure_client():
    global _azure_client
    if _azure_client is None:
        with _client_lock:
            if _azure_client is None:
                _azure_client = AzureOpenAI(
                    api_version=AZURE_API_VERSION,
                    azure_endpoint=AZURE_ENDPOINT,
                    api_key=AZURE_API_KEY,
                )
    return _azure_client


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
# 2. Azure OpenAI generation
# --------------------------------------------------------------------------

def azure_generate(prompt: str, timeout: int = 120, max_tokens: int = 4096,
                    max_retries: int = 3) -> str:
    if not AZURE_API_KEY or AZURE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        sys.exit(
            "\nYou haven't set your API key yet. Open this file and put your key "
            "between the quotes on the AZURE_API_KEY line near the top."
        )

    client = get_azure_client()
    messages = [{"role": "user", "content": prompt}]

    last_exc = None
    for attempt in range(1, max_retries + 1):
        # Try with JSON mode first; some models reject response_format, so
        # fall back to a plain call (parse_json_safely() downstream already
        # handles output that isn't perfectly strict JSON).
        for use_json_mode in (True, False):
            try:
                kwargs = {
                    "model": AZURE_DEPLOYMENT,
                    "messages": messages,
                    "max_completion_tokens": max_tokens,
                    "timeout": timeout,
                }
                if use_json_mode:
                    kwargs["response_format"] = {"type": "json_object"}
                resp = client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content
            except RateLimitError:
                # Back off longer for throttling specifically (concurrent
                # workers will hit this) rather than the generic short retry.
                wait = min(10 * attempt, 60)
                time.sleep(wait)
                last_exc = RateLimitError("rate limited")
                continue
            except Exception as e:
                msg = str(e)
                if use_json_mode and (
                    "response_format" in msg or "400" in msg or "unsupported" in msg.lower()
                ):
                    # retry immediately without json mode, same attempt
                    continue
                last_exc = e
                break
        if attempt < max_retries:
            time.sleep(3)
    raise last_exc


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
    parser = argparse.ArgumentParser(description="Build a concept wiki from every .txt file in a folder, using Azure OpenAI.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR,
                         help=f"Folder containing extracted .txt source files (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--wiki-dir", default="wiki", help="Output folder for wiki pages (default: ./wiki)")
    parser.add_argument("--chunk-size", type=int, default=3500, help="Characters per chunk (default: 3500)")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--llm-timeout", type=int, default=120,
                         help="Per-call timeout in seconds for the Azure OpenAI call (default: 120)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                         help="Max output tokens per Azure call (default: 4096). Raise if pages are "
                              "getting cut off.")
    parser.add_argument("--time-budget-min", type=int, default=0,
                         help="Stop starting new chunks after this many minutes (0 = no limit, run to "
                              "completion). Already-written pages are kept either way; rerun the same "
                              "command to continue \u2014 it always resumes automatically. Default: 0.")
    parser.add_argument("--fresh", action="store_true",
                         help="Ignore any previous progress and reprocess every chunk from scratch "
                              "(normally the script always resumes automatically).")
    parser.add_argument("--test", action="store_true",
                         help="Just send a test call to Azure OpenAI and exit, to verify the endpoint/"
                              "key/deployment at the top of this file are correct.")
    parser.add_argument("--workers", type=int, default=8,
                         help="Number of chunks sent to Azure concurrently (default: 8). Raise this to "
                              "go faster if you aren't hitting rate limits; lower it if you see repeated "
                              "'rate limited' backoff messages.")
    args = parser.parse_args()

    if args.test:
        print(f"Sending test call to Azure OpenAI (deployment: '{AZURE_DEPLOYMENT}') ...")
        try:
            reply = azure_generate('Respond with exactly this JSON: {"hello": "world"}',
                                    timeout=60, max_tokens=50)
            print(f"SUCCESS. Azure replied: {reply}")
        except SystemExit:
            raise
        except Exception as e:
            print(f"FAILED: {e}")
        return

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

    print(f"[1/3] Found {len(text_files)} .txt file(s) in {input_dir}")

    # Build one combined, ordered chunk list across every file. Each entry's
    # key ("filename#chunk-0004") is unique across the whole folder, so
    # resuming tracks position across files, not just within one.
    print("[2/3] Chunking every file ...")
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

    print(f"[3/3] Extracting concepts with Azure OpenAI (deployment: '{AZURE_DEPLOYMENT}') "
          f"using {args.workers} concurrent worker(s) and writing wiki pages ...")

    done_keys = set() if args.fresh else load_progress(wiki_dir)
    if done_keys:
        print(f"      Resuming: {len(done_keys)}/{len(all_chunks)} chunks already done, skipping those.")

    total_pages_written = 0
    run_start = time.time()
    stopped_early = False
    write_lock = threading.Lock()

    remaining_chunks = [c for c in all_chunks if c["key"] not in done_keys]
    pb = ProgressBar(total=len(remaining_chunks), desc="Wiki gen ")

    def process_one_chunk(c):
        """Runs in a worker thread: only the network call + JSON parse.
        No file writes here — those happen back on the main thread so they
        stay serialized and safe across concurrent workers."""
        key = c["key"]
        prompt = extract_concepts_for_chunk(c["text"])

        parsed = None
        last_err = None
        for attempt in range(1, MAX_LLM_RETRIES + 1):
            try:
                raw = azure_generate(prompt, timeout=args.llm_timeout, max_tokens=args.max_tokens)
            except SystemExit:
                raise
            except Exception as e:
                last_err = str(e)
                time.sleep(3)
                continue

            parsed = parse_json_safely(raw)
            if parsed is not None and "pages" in parsed:
                return c, parsed, None
            last_err = "JSON parse failed"
            time.sleep(1)

        return c, None, last_err

    time_budget_hit = False
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        chunk_iter = iter(remaining_chunks)

        def submit_next():
            try:
                nc = next(chunk_iter)
                fut = executor.submit(process_one_chunk, nc)
                futures[fut] = nc
                return True
            except StopIteration:
                return False

        # prime the pool
        for _ in range(args.workers):
            if not submit_next():
                break

        while futures:
            if args.time_budget_min > 0 and not time_budget_hit:
                elapsed_min = (time.time() - run_start) / 60
                if elapsed_min >= args.time_budget_min:
                    pb.write(f"Time budget of {args.time_budget_min} min reached "
                             f"({len(done_keys)}/{len(all_chunks)} chunks done) — finishing in-flight "
                             f"chunks, then stopping. Rerun the same command to continue.")
                    time_budget_hit = True
                    stopped_early = True

            done_fut = next(as_completed(list(futures.keys())))
            c = futures.pop(done_fut)
            key = c["key"]
            source_label = key

            try:
                _c, parsed, err = done_fut.result()
            except Exception as e:
                parsed, err = None, str(e)

            with write_lock:
                if parsed is None or "pages" not in parsed:
                    pb.write(f"{key}: giving up after {MAX_LLM_RETRIES} attempts ({err}), skipping "
                             f"(rerun the same command later, this chunk will be retried).")
                else:
                    pages = parsed.get("pages", [])
                    written = []
                    for page in pages:
                        try:
                            slug, title, category, importance = write_or_merge_page(
                                pages_dir, page, c["chunk_index"], source_label)
                            total_pages_written += 1
                            written.append((slug, title, category, importance))
                            pb.write(f"{key}: wrote/merged '{title}' ({category}/{importance}) -> pages/{slug}.md")
                        except Exception as e:
                            pb.write(f"{key}: failed to write page ({page.get('title')}): {e}")

                    append_log(wiki_dir, c["chunk_index"], source_label, written)
                    done_keys.add(key)
                    save_progress(wiki_dir, done_keys)

                pb.update(1)

            if not time_budget_hit:
                submit_next()

    pb.close()
    print("Building INDEX.md ...")
    build_index(wiki_dir, pages_dir)

    print(f"\nDone. {len(list(pages_dir.glob('*.md')))} unique wiki pages in '{pages_dir}/' "
          f"({total_pages_written} page-writes total, {len(done_keys)}/{len(all_chunks)} chunks processed "
          f"across {len(text_files)} file(s)).")
    if stopped_early:
        print("Stopped early due to --time-budget-min. Rerun the exact same command to keep going — it auto-resumes.")
    print(f"Index: {wiki_dir / 'INDEX.md'}")
    print(f"Log: {wiki_dir / 'log.md'}")


if __name__ == "__main__":
    main()
