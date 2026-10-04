"""
FAST + RESUMABLE Azure OpenAI concept-wiki generator.

What this version fixes:
- Uses JSON mode when available.
- Strongly asks the model for compact JSON.
- Uses concurrent workers for speed.
- Retries transient Azure failures with backoff.
- If JSON is malformed, automatically retries with a smaller/safer request.
- If a chunk repeatedly produces an oversized/malformed response, automatically
  splits that chunk into smaller sub-chunks instead of permanently losing it.
- Saves progress after EVERY successfully processed work item.
- Failed work is never marked complete, so rerunning resumes safely.
- Writes failed raw responses to wiki_debug/ for diagnosis.
- Builds INDEX.md at the end.
- No external dependencies beyond requests + openai.

RUN:
    python wikiAzureFinal.py

FAST RUN (recommended):
    python wikiAzureFinal.py --workers 8 --chunk-size 3500 --max-tokens 8000

If your Azure deployment has a lower rate limit, use --workers 4.
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

from openai import AzureOpenAI, RateLimitError


# ============================================================
# AZURE OPENAI CONFIG
# ============================================================

AZURE_ENDPOINT = "https://ease-azure-ai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_MODEL_NAME = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"

# Prefer an environment variable:
#   Windows PowerShell:
#   $env:AZURE_OPENAI_API_KEY="your-key"
#
# Or put your key directly between the quotes below.
AZURE_API_KEY = os.getenv(
    "AZURE_OPENAI_API_KEY",
    "PASTE_YOUR_API_KEY_HERE"
)

DEFAULT_INPUT_DIR = r"C:\Users\Praveena\merlin-rag\extracted_docs"

# Bigger chunks = fewer Azure calls = faster.
# 3500 is a good balance for reliable JSON output.
DEFAULT_CHUNK_SIZE = 3500
DEFAULT_CHUNK_OVERLAP = 200

# Per Azure-call retries.
MAX_LLM_RETRIES = 3

# If a chunk repeatedly fails JSON parsing, split it automatically.
MIN_SPLIT_SIZE = 900


# ============================================================
# GLOBAL AZURE CLIENT
# ============================================================

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


# ============================================================
# TERMINAL PROGRESS
# ============================================================

class ProgressBar:
    def __init__(self, total, desc="Wiki", width=30):
        self.total = max(total, 0)
        self.desc = desc
        self.width = width
        self.count = 0
        self.start = time.time()
        self.last_len = 0

    @staticmethod
    def fmt_time(seconds):
        seconds = max(int(seconds), 0)
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)

        if h:
            return f"{h}h{m:02d}m{s:02d}s"
        if m:
            return f"{m}m{s:02d}s"
        return f"{s}s"

    def update(self, n=1):
        self.count += n
        self.render()

    def write(self, message):
        sys.stdout.write("\r" + (" " * self.last_len) + "\r")
        print(message)
        self.render()

    def render(self):
        if self.total <= 0:
            return

        fraction = min(self.count / self.total, 1.0)
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)

        elapsed = time.time() - self.start
        rate = self.count / elapsed if elapsed > 0 else 0
        remaining = (
            (self.total - self.count) / rate
            if rate > 0
            else 0
        )

        line = (
            f"\r{self.desc} [{bar}] "
            f"{self.count}/{self.total} "
            f"({fraction * 100:5.1f}%) "
            f"elapsed {self.fmt_time(elapsed)} "
            f"eta {self.fmt_time(remaining)}"
        )

        padding = max(self.last_len - len(line), 0)
        sys.stdout.write(line + (" " * padding))
        sys.stdout.flush()
        self.last_len = len(line)

    def close(self):
        self.render()
        sys.stdout.write("\n")
        sys.stdout.flush()


# ============================================================
# FILE READING + CHUNKING
# ============================================================

def read_text_file(path):
    return path.read_text(encoding="utf-8", errors="ignore")


def find_text_files(input_dir):
    return sorted(input_dir.glob("*.txt"))


def chunk_text(text, chunk_size=DEFAULT_CHUNK_SIZE, overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Paragraph-aware chunking with overlap.
    """
    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (
                f"{current}\n\n{para}"
                if current
                else para
            )
        else:
            if current:
                chunks.append(current)

            if len(para) > chunk_size:
                start = 0

                while start < len(para):
                    chunks.append(
                        para[start:start + chunk_size]
                    )
                    start += max(1, chunk_size - overlap)

                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Add overlap between chunks.
    overlapped = []

    for i, chunk in enumerate(chunks):
        if i == 0 or not overlap:
            overlapped.append(chunk)
        else:
            tail = chunks[i - 1][-overlap:]
            overlapped.append(
                tail + "\n\n" + chunk
            )

    return overlapped


def split_chunk_text(text):
    """
    Emergency splitter used when a model repeatedly fails to return valid JSON.
    """
    if len(text) <= MIN_SPLIT_SIZE:
        return [text]

    midpoint = len(text) // 2

    # Prefer a paragraph boundary.
    candidates = [
        text.rfind("\n\n", 0, midpoint),
        text.find("\n\n", midpoint),
        text.rfind(". ", 0, midpoint),
        text.find(". ", midpoint),
    ]

    valid = [
        p for p in candidates
        if p >= max(100, int(len(text) * 0.30))
        and p <= int(len(text) * 0.70)
    ]

    split_at = min(
        valid,
        key=lambda p: abs(p - midpoint)
    ) if valid else midpoint

    left = text[:split_at].strip()
    right = text[split_at:].strip()

    if not left or not right:
        return [text]

    return [left, right]


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are maintaining a persistent Rolls-Royce SMR engineering and regulatory
concept wiki.

Extract meaningful concepts from the supplied source text.

IMPORTANT:
- Return ONLY valid JSON.
- Do NOT use markdown fences.
- Do NOT write anything before or after the JSON.
- Do NOT invent facts.
- Use exact source terminology where possible.
- Prefer useful concepts over trivial words.
- Create separate pages for named systems, components, requirements,
  hazards, processes, documents, organisations, claims, evidence,
  standards and assessment topics.
- Preserve acronyms exactly.
- Keep the output compact enough to fit the response limit.

Return EXACTLY:

{
  "pages": [
    {
      "title": "Page Title",
      "slug": "page-slug",
      "category": "System",
      "importance": "Important",
      "aliases": [],
      "relationships": {
        "related_to": [],
        "part_of": [],
        "contains": [],
        "depends_on": [],
        "interfaces_with": [],
        "regulated_by": [],
        "references": []
      },
      "content": "FULL MARKDOWN PAGE"
    }
  ]
}

Allowed categories:
Reactor, System, Subsystem, Component, Structure, Equipment,
Material, Fuel, Process, Procedure, Requirement, Safety, Security,
Safeguards, Environment, Regulation, Standard, Document, Organisation,
Regulator, Assessment, Hazard, Fault, Accident, Risk, Claim, Evidence,
Interface, Assumption, Constraint, Operation, Maintenance, Inspection,
Test, Waste, RadiologicalProtection, QualityAssurance, Programme, Other.

Importance:
Critical = central to nuclear safety, licensing, reactor design or major safety functions.
Important = materially relevant to engineering, assessment, compliance or operation.
Reference = supporting/background detail.

The markdown content should contain:

---
title: Page Title
slug: page-slug
category: System
importance: Important
aliases: []
systems: []
components: []
related: []
sources: []
status: Draft
tags: []
---

# Page Title

## Summary
1-3 concise sentences.

## Definition
Definition grounded only in the source.

## Technical Details
Important details supported by the source.

## Role in Rolls-Royce SMR
Only if supported by the source.

## Safety, Environmental, Security, or Safeguards Relevance
Only if supported by the source.

## Relationships
Use [[slug]] links only for relationships supported by the source.

## Open Points
Only uncertainties/open issues supported by the source.

## Sources
- Source text chunk

If no meaningful concept exists:
{"pages":[]}
"""


def build_prompt(chunk_text_value, compact=False):
    extra = ""

    if compact:
        extra = """
COMPACT RETRY:
The previous response could not be parsed as JSON.
Return fewer, high-value concept pages and keep each content field concise.
Never truncate JSON. If necessary, return fewer pages rather than producing
incomplete JSON.
"""

    return (
        SYSTEM_PROMPT
        + extra
        + "\nSOURCE TEXT:\n"
        + chunk_text_value
    )


# ============================================================
# AZURE CALL
# ============================================================

def azure_generate(prompt, timeout=120, max_tokens=8000):
    """
    Calls Azure OpenAI.

    JSON mode is attempted first.
    If the deployment rejects response_format, the call is retried
    without it.
    """

    if not AZURE_API_KEY or AZURE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise RuntimeError(
            "Azure API key is not configured. Set AZURE_OPENAI_API_KEY "
            "or put the key in AZURE_API_KEY."
        )

    client = get_azure_client()

    last_error = None

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            kwargs = {
                "model": AZURE_DEPLOYMENT,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "max_completion_tokens": max_tokens,
                "timeout": timeout,
            }

            try:
                kwargs["response_format"] = {
                    "type": "json_object"
                }

                response = client.chat.completions.create(
                    **kwargs
                )

            except Exception as json_mode_error:
                msg = str(json_mode_error).lower()

                # Some Azure/model combinations reject JSON mode.
                if (
                    "response_format" not in msg
                    and "unsupported" not in msg
                    and "400" not in msg
                    and "json" not in msg
                ):
                    raise

                kwargs.pop("response_format", None)

                response = client.chat.completions.create(
                    **kwargs
                )

            content = response.choices[0].message.content

            if not content:
                raise RuntimeError(
                    "Azure returned an empty response."
                )

            return content

        except RateLimitError as exc:
            last_error = exc

            # Exponential-ish backoff.
            wait = min(5 * attempt, 30)
            time.sleep(wait)

        except Exception as exc:
            last_error = exc

            if attempt < MAX_LLM_RETRIES:
                time.sleep(2 * attempt)

    raise last_error


# ============================================================
# JSON PARSING
# ============================================================

def parse_json_safely(raw):
    """
    Robust JSON parser.

    Handles:
    - normal JSON
    - ```json fences
    - accidental text around JSON
    - raw control characters
    """

    if not raw:
        return None, "empty response"

    text = raw.strip()

    # Remove markdown fences.
    text = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    text = text.strip()

    candidates = [text]

    # Find outer JSON object.
    first = text.find("{")
    last = text.rfind("}")

    if first >= 0 and last > first:
        candidate = text[first:last + 1]

        if candidate not in candidates:
            candidates.append(candidate)

    errors = []

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)

            if isinstance(parsed, dict) and "pages" in parsed:
                return parsed, None

        except json.JSONDecodeError as exc:
            errors.append(
                f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
            )

        try:
            parsed = json.loads(
                candidate,
                strict=False,
            )

            if isinstance(parsed, dict) and "pages" in parsed:
                return parsed, None

        except json.JSONDecodeError as exc:
            errors.append(
                f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
            )

    return None, "; ".join(errors[-2:]) or "invalid JSON"


# ============================================================
# SLUGGING
# ============================================================

def slugify(text):
    text = (
        unicodedata
        .normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode()
    )

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text)

    return text.strip("-")


# ============================================================
# PROGRESS
# ============================================================

def load_key_set(path):
    if not path.exists():
        return set()

    try:
        data = json.loads(
            path.read_text(encoding="utf-8")
        )

        return set(data)

    except Exception:
        return set()


def save_key_set(path, keys):
    temp = path.with_suffix(".tmp")

    temp.write_text(
        json.dumps(
            sorted(keys),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Atomic replacement prevents a half-written progress file.
    temp.replace(path)


# ============================================================
# MARKDOWN FRONTMATTER
# ============================================================

FRONTMATTER_RE = re.compile(
    r"^---\n(.*?)\n---\n(.*)$",
    re.DOTALL,
)

LIST_FIELD_RE = re.compile(
    r"^(\w+):\s*\[(.*)\]\s*$"
)


def split_frontmatter(md_text):
    m = FRONTMATTER_RE.match(
        md_text.strip() + "\n"
    )

    if not m:
        return {}, md_text

    fm_raw, body = m.group(1), m.group(2)

    fields = {}

    for line in fm_raw.splitlines():
        line = line.strip()

        if not line or ":" not in line:
            continue

        list_match = LIST_FIELD_RE.match(line)

        if list_match:
            key, items = list_match.groups()

            fields[key] = [
                item.strip().strip('"').strip("'")
                for item in items.split(",")
                if item.strip()
            ]

        else:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()

    return fields, body.strip()


def render_frontmatter(fields):
    lines = ["---"]

    order = [
        "title",
        "slug",
        "category",
        "importance",
        "aliases",
        "systems",
        "components",
        "related",
        "sources",
        "status",
        "tags",
    ]

    seen = set()

    for key in order:
        if key not in fields:
            continue

        seen.add(key)
        value = fields[key]

        if isinstance(value, list):
            lines.append(
                f"{key}: [{', '.join(map(str, value))}]"
            )
        else:
            lines.append(
                f"{key}: {value}"
            )

    for key, value in fields.items():
        if key in seen:
            continue

        if isinstance(value, list):
            lines.append(
                f"{key}: [{', '.join(map(str, value))}]"
            )
        else:
            lines.append(
                f"{key}: {value}"
            )

    lines.append("---")

    return "\n".join(lines)


def merge_list_field(old_list, new_list):
    merged = list(old_list or [])

    for item in new_list or []:
        if item not in merged:
            merged.append(item)

    return merged


# ============================================================
# PAGE WRITING
# ============================================================

def write_or_merge_page(
    pages_dir,
    page,
    source_label,
):
    title = str(
        page.get("title")
        or "Untitled Concept"
    ).strip()

    slug = page.get("slug") or slugify(title)
    slug = slugify(str(slug))

    if not slug:
        slug = "untitled-concept"

    out_path = pages_dir / f"{slug}.md"

    new_content = str(
        page.get("content") or ""
    ).strip()

    # If the model forgot frontmatter, create safe content.
    new_fields, new_body = split_frontmatter(
        new_content
    )

    if not new_body:
        new_body = (
            f"# {title}\n\n"
            "## Summary\n"
            "Concept extracted from the source text.\n\n"
            "## Sources\n"
            f"- {source_label}\n"
        )

    # Ensure required fields exist.
    new_fields.setdefault("title", title)
    new_fields.setdefault("slug", slug)
    new_fields.setdefault(
        "category",
        page.get("category", "Other"),
    )
    new_fields.setdefault(
        "importance",
        page.get("importance", "Reference"),
    )
    new_fields.setdefault(
        "status",
        "Draft",
    )

    relationships = page.get(
        "relationships",
        {},
    )

    aliases = new_fields.get("aliases", [])
    systems = new_fields.get("systems", [])
    components = new_fields.get("components", [])
    related = new_fields.get("related", [])
    sources = new_fields.get("sources", [])

    if not isinstance(aliases, list):
        aliases = []

    if not isinstance(systems, list):
        systems = []

    if not isinstance(components, list):
        components = []

    if not isinstance(related, list):
        related = []

    if not isinstance(sources, list):
        sources = []

    # Add relationship slugs to related.
    for rel_type in (
        "related_to",
        "part_of",
        "contains",
        "depends_on",
        "interfaces_with",
        "regulated_by",
        "references",
    ):
        values = relationships.get(
            rel_type,
            [],
        )

        if isinstance(values, list):
            related.extend(
                str(v)
                for v in values
                if v
            )

    new_fields["aliases"] = merge_list_field(
        aliases,
        page.get("aliases", []),
    )

    new_fields["systems"] = systems
    new_fields["components"] = components

    new_fields["related"] = list(
        dict.fromkeys(related)
    )

    new_fields["sources"] = merge_list_field(
        sources,
        [source_label],
    )

    # Keep the strongest importance.
    importance_rank = {
        "Critical": 3,
        "Important": 2,
        "Reference": 1,
    }

    new_importance = new_fields.get(
        "importance",
        "Reference",
    )

    if out_path.exists():
        existing = out_path.read_text(
            encoding="utf-8"
        )

        old_fields, old_body = split_frontmatter(
            existing
        )

        merged_fields = dict(old_fields)

        for key in (
            "aliases",
            "systems",
            "components",
            "related",
            "sources",
            "tags",
        ):
            merged_fields[key] = merge_list_field(
                old_fields.get(key, []),
                new_fields.get(key, []),
            )

        old_importance = old_fields.get(
            "importance",
            "Reference",
        )

        if (
            importance_rank.get(
                new_importance,
                1,
            )
            > importance_rank.get(
                old_importance,
                1,
            )
        ):
            merged_fields["importance"] = (
                new_importance
            )

        for key in (
            "title",
            "slug",
            "category",
            "status",
        ):
            merged_fields[key] = (
                old_fields.get(key)
                or new_fields.get(key)
            )

        # Avoid duplicate H1.
        new_body = re.sub(
            r"^#\s+.*\n+",
            "",
            new_body,
            count=1,
        )

        combined_body = (
            f"{old_body}\n\n"
            "---\n\n"
            f"## Additional Extraction ({source_label})\n\n"
            f"{new_body}"
        )

        final_text = (
            render_frontmatter(
                merged_fields
            )
            + "\n\n"
            + combined_body.strip()
            + "\n"
        )

    else:
        final_text = (
            render_frontmatter(
                new_fields
            )
            + "\n\n"
            + new_body.strip()
            + "\n"
        )

    # Atomic page write.
    temp_path = out_path.with_suffix(".tmp")

    temp_path.write_text(
        final_text,
        encoding="utf-8",
    )

    temp_path.replace(out_path)

    return (
        slug,
        new_fields.get(
            "title",
            title,
        ),
        new_fields.get(
            "category",
            "Other",
        ),
        new_fields.get(
            "importance",
            "Reference",
        ),
    )


# ============================================================
# INDEX
# ============================================================

def extract_summary_lines(md_text):
    match = re.search(
        r"##\s*Summary\s*\n(.*?)(?=\n##|\Z)",
        md_text,
        re.DOTALL,
    )

    if not match:
        return [
            "No summary extracted.",
            "See the full page for details.",
        ]

    raw = " ".join(
        match.group(1)
        .strip()
        .split()
    )

    if not raw:
        return [
            "No summary extracted.",
            "See the full page for details.",
        ]

    sentences = [
        s.strip()
        for s in re.split(
            r"(?<=[.!?])\s+",
            raw,
        )
        if s.strip()
    ]

    if len(sentences) >= 2:
        return sentences[:2]

    return [
        sentences[0],
        "See the full page for further detail and sources.",
    ]


def build_index(wiki_dir, pages_dir):
    pages = []

    for path in sorted(
        pages_dir.glob("*.md")
    ):
        try:
            text = path.read_text(
                encoding="utf-8"
            )

            fields, _ = split_frontmatter(
                text
            )

            title = fields.get(
                "title",
                path.stem,
            )

            pages.append(
                {
                    "slug": path.stem,
                    "title": title,
                    "summary_lines": extract_summary_lines(
                        text
                    ),
                }
            )

        except Exception:
            continue

    pages.sort(
        key=lambda p: p["title"].lower()
    )

    lines = [
        "# Rolls-Royce SMR Wiki",
        "",
        "## Concept Pages",
        "",
    ]

    for page in pages:
        lines.append(
            f"* [[{page['slug']}]] — {page['title']}"
        )

        for summary in page["summary_lines"]:
            lines.append(
                f"  {summary}"
            )

        lines.append("")

    index_path = wiki_dir / "INDEX.md"
    temp_path = wiki_dir / "INDEX.tmp"

    temp_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    temp_path.replace(index_path)


# ============================================================
# LOG
# ============================================================

def append_log(
    wiki_dir,
    source_label,
    written,
):
    log_path = wiki_dir / "log.md"

    if not log_path.exists():
        log_path.write_text(
            "# Wiki Ingestion Log\n\n",
            encoding="utf-8",
        )

    timestamp = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    lines = [
        f"## {timestamp} — {source_label}",
        "",
    ]

    if written:
        for slug, title, category, importance in written:
            lines.append(
                f"- [{title}](pages/{slug}.md) "
                f"({category}/{importance})"
            )
    else:
        lines.append(
            "- (no pages written)"
        )

    lines.append("")

    with log_path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n".join(lines) + "\n"
        )


# ============================================================
# DEBUG OUTPUT
# ============================================================

def save_debug(
    debug_dir,
    key,
    raw,
    attempt,
    error,
):
    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_key = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        key,
    )

    path = (
        debug_dir
        / f"{safe_key}_attempt{attempt}.txt"
    )

    content = (
        f"ERROR:\n{error}\n\n"
        "RAW RESPONSE:\n"
        + (raw or "(empty)")
    )

    try:
        path.write_text(
            content,
            encoding="utf-8",
        )
    except Exception:
        pass


# ============================================================
# SINGLE WORK ITEM
# ============================================================

def process_text_recursive(
    key,
    file_name,
    chunk_index,
    text,
    args,
    debug_dir,
    depth=0,
):
    """
    Try a chunk several times.

    If the model repeatedly returns invalid JSON, split the chunk into
    two smaller pieces and process both recursively.

    IMPORTANT:
    The parent chunk is considered successful only when ALL child pieces
    succeed. This means the progress checkpoint remains safe across
    restarts and no content is silently lost.
    """

    last_error = None

    # Prevent unlimited recursion.
    max_depth = 6

    for attempt in range(1, MAX_LLM_RETRIES + 1):
        compact = attempt == MAX_LLM_RETRIES

        try:
            prompt = build_prompt(
                text,
                compact=compact,
            )

            raw = azure_generate(
                prompt,
                timeout=args.llm_timeout,
                max_tokens=args.max_tokens,
            )

            parsed, parse_error = parse_json_safely(raw)

            if parsed is not None:
                pages = parsed.get("pages", [])

                if not isinstance(pages, list):
                    raise ValueError(
                        '"pages" is not a list'
                    )

                return {
                    "success": True,
                    "pages": pages,
                    "error": None,
                    "splits": depth,
                }

            last_error = (
                f"JSON parse failed: {parse_error}"
            )

            save_debug(
                debug_dir,
                key,
                raw,
                attempt,
                last_error,
            )

        except Exception as exc:
            last_error = str(exc)

            save_debug(
                debug_dir,
                key,
                "",
                attempt,
                last_error,
            )

        if attempt < MAX_LLM_RETRIES:
            time.sleep(attempt)

    # --------------------------------------------------------
    # Automatic emergency split
    # --------------------------------------------------------

    if (
        len(text) > MIN_SPLIT_SIZE
        and depth < max_depth
    ):
        pieces = split_chunk_text(text)

        if len(pieces) == 2:
            all_pages = []
            total_splits = 1

            for part_number, piece in enumerate(pieces, 1):
                child_key = (
                    f"{key}.part{part_number}"
                )

                child_result = process_text_recursive(
                    child_key,
                    file_name,
                    chunk_index,
                    piece,
                    args,
                    debug_dir,
                    depth=depth + 1,
                )

                if not child_result["success"]:
                    return {
                        "success": False,
                        "pages": [],
                        "error": (
                            f"{key}: child "
                            f"{child_key} failed: "
                            f"{child_result['error']}"
                        ),
                        "splits": total_splits
                        + child_result.get(
                            "splits",
                            0,
                        ),
                    }

                all_pages.extend(
                    child_result["pages"]
                )

                total_splits += child_result.get(
                    "splits",
                    0,
                )

            return {
                "success": True,
                "pages": all_pages,
                "error": None,
                "splits": total_splits,
            }

    return {
        "success": False,
        "pages": [],
        "error": last_error
        or "Unknown processing error",
        "splits": depth,
    }


def process_work_item(
    item,
    args,
    debug_dir,
):
    return process_text_recursive(
        key=item["key"],
        file_name=item["file_name"],
        chunk_index=item["chunk_index"],
        text=item["text"],
        args=args,
        debug_dir=debug_dir,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fast, resumable Azure OpenAI concept wiki generator."
        )
    )

    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
    )

    parser.add_argument(
        "--wiki-dir",
        default="wiki",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
    )

    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8000,
        help=(
            "Maximum completion tokens. "
            "8000 is normally enough and avoids unnecessarily "
            "large responses."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Concurrent Azure calls. "
            "8 is recommended; lower to 4 if rate limits occur."
        ),
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore previous progress.",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Test Azure connection and exit.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Validate config
    # --------------------------------------------------------

    if not AZURE_API_KEY or (
        AZURE_API_KEY
        == "PASTE_YOUR_API_KEY_HERE"
    ):
        print(
            "\nERROR: Azure API key is not configured.\n"
            "Set AZURE_OPENAI_API_KEY or put your key "
            "in AZURE_API_KEY near the top of this file.\n"
        )

        sys.exit(1)

    if args.workers < 1:
        args.workers = 1

    # --------------------------------------------------------
    # Azure test
    # --------------------------------------------------------

    if args.test:
        print(
            f"Testing Azure deployment "
            f"'{AZURE_DEPLOYMENT}'..."
        )

        try:
            raw = azure_generate(
                (
                    'Return exactly this JSON and nothing else: '
                    '{"hello":"world"}'
                ),
                timeout=60,
                max_tokens=50,
            )

            parsed, error = parse_json_safely(
                raw
            )

            if parsed is not None:
                print(
                    "SUCCESS:",
                    json.dumps(parsed),
                )
            else:
                print(
                    "Azure responded, but JSON parsing failed:"
                )
                print(raw)
                print(error)

        except Exception as exc:
            print(
                "AZURE TEST FAILED:",
                repr(exc),
            )
            sys.exit(1)

        return

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(
            f"Input folder not found: {input_dir}"
        )
        sys.exit(1)

    text_files = find_text_files(
        input_dir
    )

    if not text_files:
        print(
            f"No .txt files found in {input_dir}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    wiki_dir = Path(args.wiki_dir)
    pages_dir = wiki_dir / "pages"
    debug_dir = Path("wiki_debug")

    pages_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    progress_path = (
        wiki_dir / ".progress.json"
    )

    # --------------------------------------------------------
    # Build chunks
    # --------------------------------------------------------

    print(
        f"\n[1/3] Found {len(text_files)} "
        f".txt file(s)"
    )

    all_items = []

    print("[2/3] Chunking files...")

    for file_path in text_files:
        text = read_text_file(
            file_path
        )

        chunks = chunk_text(
            text,
            args.chunk_size,
            args.chunk_overlap,
        )

        for index, chunk in enumerate(chunks):
            all_items.append(
                {
                    "key": (
                        f"{file_path.name}"
                        f"#chunk-{index:04d}"
                    ),
                    "file_name": file_path.name,
                    "chunk_index": index,
                    "text": chunk,
                }
            )

        print(
            f"      {file_path.name}: "
            f"{len(chunks)} chunks"
        )

    print(
        f"      TOTAL: {len(all_items)} chunks"
    )

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    done_keys = (
        set()
        if args.fresh
        else load_key_set(
            progress_path
        )
    )

    remaining = [
        item
        for item in all_items
        if item["key"] not in done_keys
    ]

    print(
        f"\n[3/3] Azure extraction"
    )

    print(
        f"      Deployment: {AZURE_DEPLOYMENT}"
    )

    print(
        f"      Workers: {args.workers}"
    )

    print(
        f"      Chunk size: {args.chunk_size}"
    )

    print(
        f"      Remaining: "
        f"{len(remaining)}/{len(all_items)}"
    )

    if not remaining:
        print(
            "\nEverything is already processed."
        )

        build_index(
            wiki_dir,
            pages_dir,
        )

        print(
            f"Index: {wiki_dir / 'INDEX.md'}"
        )

        return

    # --------------------------------------------------------
    # Threaded processing
    # --------------------------------------------------------

    pb = ProgressBar(
        len(remaining),
        desc="Wiki gen",
    )

    start_time = time.time()
    total_page_writes = 0
    split_count = 0
    failed_count = 0

    # Each future represents one ORIGINAL chunk.
    # A future internally handles emergency splitting if required.
    with ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:

        futures = {
            executor.submit(
                process_work_item,
                item,
                args,
                debug_dir,
            ): item
            for item in remaining
        }

        for future in as_completed(futures):
            item = futures[future]
            key = item["key"]

            try:
                result = future.result()

            except Exception as exc:
                result = {
                    "success": False,
                    "pages": [],
                    "error": str(exc),
                    "splits": 0,
                }

            # ------------------------------------------------
            # Successful result
            # ------------------------------------------------

            if result["success"]:
                pages = result["pages"]
                written = []

                try:
                    for page in pages:
                        if not isinstance(page, dict):
                            continue

                        try:
                            info = write_or_merge_page(
                                pages_dir,
                                page,
                                key,
                            )

                            written.append(info)
                            total_page_writes += 1

                        except Exception as page_error:
                            pb.write(
                                f"{key}: page write failed: "
                                f"{page_error}"
                            )

                    append_log(
                        wiki_dir,
                        key,
                        written,
                    )

                    # ONLY after page writing + log writing,
                    # checkpoint the original chunk.
                    done_keys.add(key)
                    save_key_set(
                        progress_path,
                        done_keys,
                    )

                    split_count += result.get(
                        "splits",
                        0,
                    )

                    pb.write(
                        f"{key}: "
                        f"{len(written)} page(s)"
                        + (
                            f", auto-split "
                            f"{result['splits']}x"
                            if result.get("splits", 0)
                            else ""
                        )
                    )

                except Exception as exc:
                    # Do NOT checkpoint this chunk.
                    failed_count += 1

                    pb.write(
                        f"{key}: checkpoint/write failure: "
                        f"{exc} — rerun to retry"
                    )

            # ------------------------------------------------
            # Failed result
            # ------------------------------------------------

            else:
                failed_count += 1

                pb.write(
                    f"{key}: FAILED after retries: "
                    f"{result.get('error')} "
                    f"— NOT marked complete"
                )

            pb.update(1)

    pb.close()

    # --------------------------------------------------------
    # Final index
    # --------------------------------------------------------

    print(
        "\nBuilding INDEX.md..."
    )

    build_index(
        wiki_dir,
        pages_dir,
    )

    elapsed = (
        time.time() - start_time
    )

    final_done = len(done_keys)

    total_pages = len(
        list(
            pages_dir.glob("*.md")
        )
    )

    print("\n" + "=" * 60)
    print("WIKI GENERATION FINISHED")
    print("=" * 60)

    print(
        f"Unique wiki pages : {total_pages}"
    )

    print(
        f"Page writes       : {total_page_writes}"
    )

    print(
        f"Completed chunks  : "
        f"{final_done}/{len(all_items)}"
    )

    print(
        f"Emergency splits  : {split_count}"
    )

    print(
        f"Failures           : {failed_count}"
    )

    print(
        f"Elapsed            : "
        f"{ProgressBar.fmt_time(elapsed)}"
    )

    print(
        f"Pages folder       : {pages_dir}"
    )

    print(
        f"Index              : "
        f"{wiki_dir / 'INDEX.md'}"
    )

    print(
        f"Debug              : {debug_dir}"
    )

    if failed_count:
        print(
            "\nSome chunks failed and were NOT marked complete."
        )
        print(
            "Run the exact same command again. "
            "The progress file will resume safely."
        )
    else:
        print(
            "\nALL ORIGINAL CHUNKS ARE COMPLETE."
        )


if __name__ == "__main__":
    main()
