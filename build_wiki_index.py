from pathlib import Path
import re
import chromadb
from sentence_transformers import SentenceTransformer

WIKI_DIR = Path("wiki")
PAGES_DIR = WIKI_DIR / "pages"
CHROMA_DIR = Path("./wiki_chroma_store")
COLLECTION_NAME = "wikillm-index"
MODEL_NAME = "all-mpnet-base-v2"
BATCH_SIZE = 64


def parse_frontmatter(text):
    if not text.startswith("---"):
        return {}
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}
    fields = {}
    for line in parts[1].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip()] = v.strip()
    return fields


def index_text(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    fm = parse_frontmatter(text)
    title = fm.get("title", path.stem)
    # Index the complete page so semantic retrieval sees definitions,
    # technical details and relationships, while keeping the source page intact.
    return f"TITLE: {title}\nSLUG: {path.stem}\n\n{text}"


def main():
    if not PAGES_DIR.exists():
        raise SystemExit(f"Wiki pages folder not found: {PAGES_DIR}")
    pages = sorted(PAGES_DIR.glob("*.md"))
    if not pages:
        raise SystemExit(f"No markdown pages found in {PAGES_DIR}")

    print(f"Found {len(pages)} wiki pages.")
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    documents = [index_text(p) for p in pages]
    ids = [p.stem for p in pages]
    metadatas = [{"slug": p.stem, "title": p.stem.replace("-", " ")} for p in pages]

    print("Creating embeddings...")
    for start in range(0, len(documents), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(documents))
        embeddings = model.encode(
            documents[start:end],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            embeddings=embeddings,
        )
        print(f"Indexed {end}/{len(documents)}")

    print("\n========================================")
    print("WIKI INDEX READY")
    print("========================================")
    print(f"Wiki pages : {len(pages)}")
    print(f"Index rows  : {collection.count()}")
    print(f"Chroma path : {CHROMA_DIR}")


if __name__ == "__main__":
    main()
