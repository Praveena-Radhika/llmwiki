import re
import hashlib
from pathlib import Path
from collections import Counter
import chromadb


# ============================================================
# CONFIG
# ============================================================

INPUT_DIR = Path(r"C:\Users\Praveena\merlin-rag\extracted_docs")
CHROMA_DIR = Path("./chroma_store")
COLLECTION_NAME = "extracted-doc"

CHUNK_SIZE = 1800
CHUNK_OVERLAP = 200


# ============================================================
# SAME CHUNKER USED BY WIKI PIPELINE
# ============================================================

def read_text_file(path):
    return path.read_text(
        encoding="utf-8",
        errors="ignore"
    )


def chunk_text(text, chunk_size=1800, overlap=200):
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
                    start += chunk_size - overlap

                current = ""
            else:
                current = para

    if current:
        chunks.append(current)

    # Same overlap logic as Wiki pipeline
    overlapped = []

    for i, chunk in enumerate(chunks):

        if i == 0:
            overlapped.append(chunk)

        else:
            tail = (
                chunks[i - 1][-overlap:]
                if overlap
                else ""
            )

            overlapped.append(
                (tail + "\n\n" + chunk)
                if tail
                else chunk
            )

    return overlapped


# ============================================================
# BUILD EXPECTED CHUNKS
# ============================================================

print("\n========================================")
print("RAG PREPARATION")
print("========================================")

print("\n[1/5] Reading source documents...")

files = sorted(INPUT_DIR.glob("*.txt"))

if not files:
    raise SystemExit(
        f"No .txt files found in {INPUT_DIR}"
    )

print(f"Found {len(files)} source documents.")

expected = {}

for file in files:

    text = read_text_file(file)

    chunks = chunk_text(
        text,
        CHUNK_SIZE,
        CHUNK_OVERLAP
    )

    for i, chunk in enumerate(chunks):

        key = (
            file.name,
            i
        )

        expected[key] = chunk

print(
    f"Expected chunks: {len(expected)}"
)


# ============================================================
# LOAD CHROMA
# ============================================================

print("\n[2/5] Loading Chroma...")

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    COLLECTION_NAME
)

print(
    f"Existing vectors: {collection.count()}"
)


# ============================================================
# READ EXISTING VECTORS
# ============================================================

print("\n[3/5] Checking existing vectors...")

existing = collection.get(
    include=[
        "documents",
        "metadatas"
    ]
)

existing_keys = set()
mismatches = []

for document, metadata in zip(
    existing["documents"],
    existing["metadatas"]
):

    if not metadata:
        continue

    source_file = metadata.get(
        "source_file"
    )

    chunk_index = metadata.get(
        "chunk_index"
    )

    if source_file is None or chunk_index is None:
        mismatches.append(
            "Vector with missing metadata"
        )
        continue

    key = (
        source_file,
        int(chunk_index)
    )

    existing_keys.add(key)

    # CRITICAL:
    # Verify that the stored text is exactly
    # the same chunk produced by our Wiki chunker.

    expected_text = expected.get(key)

    if expected_text is None:

        mismatches.append(
            f"Unexpected Chroma chunk: {key}"
        )

    elif document != expected_text:

        mismatches.append(
            f"TEXT MISMATCH: {source_file} "
            f"chunk {chunk_index}"
        )


print(
    f"Existing chunk metadata entries: "
    f"{len(existing_keys)}"
)

print(
    f"Exact text mismatches: "
    f"{len(mismatches)}"
)


# ============================================================
# SAFETY CHECK
# ============================================================

if mismatches:

    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("STOP — CHUNK ALIGNMENT PROBLEM")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(
        "\nThe existing Chroma vectors do NOT exactly "
        "match the Wiki chunking."
    )

    print("\nFirst problems:")

    for item in mismatches[:20]:
        print(" -", item)

    print(
        "\nNO vectors were modified."
    )

    print(
        "\nSend me this output before doing anything else."
    )

    raise SystemExit(1)


# ============================================================
# FIND MISSING CHUNKS
# ============================================================

missing = [
    key
    for key in expected
    if key not in existing_keys
]

print(
    f"\nMissing vectors: {len(missing)}"
)


# ============================================================
# REPORT BY DOCUMENT
# ============================================================

missing_by_file = Counter(
    key[0]
    for key in missing
)

if missing_by_file:

    print("\nMissing vectors by document:")

    for filename, count in sorted(
        missing_by_file.items()
    ):
        print(
            f"{count:4} -> {filename}"
        )


# ============================================================
# NOTHING TO ADD
# ============================================================

if not missing:

    print(
        "\n========================================"
    )
    print(
        "ALL CHUNKS ARE ALREADY EMBEDDED."
    )
    print(
        "========================================"
    )

    print(
        f"Vectors: {collection.count()}"
    )

    raise SystemExit(0)


# ============================================================
# CHECK EMBEDDING FUNCTION
# ============================================================

print(
    "\n[4/5] Checking Chroma embedding function..."
)

embedding_function = getattr(
    collection,
    "_embedding_function",
    None
)

print(
    "Embedding function:",
    type(embedding_function).__name__
    if embedding_function
    else "NOT AVAILABLE"
)


if embedding_function is None:

    print(
        "\nChroma does not expose the original "
        "embedding function."
    )

    print(
        "NO vectors were modified."
    )

    print(
        "\nWe need to use the original embedding "
        "configuration to add the missing chunks."
    )

    raise SystemExit(2)


# ============================================================
# ADD ONLY MISSING CHUNKS
# ============================================================

print(
    "\n[5/5] Adding ONLY missing vectors..."
)

documents = []
metadatas = []
ids = []

for source_file, chunk_index in missing:

    text = expected[
        (source_file, chunk_index)
    ]

    documents.append(text)

    metadatas.append({
        "source_file": source_file,
        "chunk_index": chunk_index
    })

    # Deterministic ID
    raw_id = (
        f"{source_file}::chunk-{chunk_index}"
    )

    stable_id = (
        "rag-" +
        hashlib.sha1(
            raw_id.encode("utf-8")
        ).hexdigest()
    )

    ids.append(stable_id)


# Batch additions for speed
BATCH_SIZE = 100

for start in range(
    0,
    len(documents),
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        len(documents)
    )

    collection.add(
        documents=documents[start:end],
        metadatas=metadatas[start:end],
        ids=ids[start:end]
    )

    print(
        f"Added {end}/{len(documents)}"
    )


# ============================================================
# FINAL CHECK
# ============================================================

final_count = collection.count()

print(
    "\n========================================"
)

print(
    "RAG PREPARATION COMPLETE"
)

print(
    "========================================"
)

print(
    f"Expected chunks : {len(expected)}"
)

print(
    f"Final vectors   : {final_count}"
)

print(
    f"Added           : {len(missing)}"
)

if final_count == len(expected):

    print(
        "\nSUCCESS:"
    )

    print(
        "Chroma now contains the complete "
        "2,981-chunk corpus."
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "Vector count does not match expected "
        "chunk count."
    )