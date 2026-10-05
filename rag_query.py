import chromadb
from sentence_transformers import SentenceTransformer
from openai import AzureOpenAI

# ============================================================
# CONFIG
# ============================================================

CHROMA_DIR = "./chroma_store"
COLLECTION_NAME = "extracted-doc"

# Same Azure configuration used by your Wiki pipeline
AZURE_ENDPOINT = "https://ease-azure-ai.openai.azure.com/"
AZURE_DEPLOYMENT = "gpt-5.4"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_API_KEY = "PASTE_YOUR_API_KEY_HERE"

# IMPORTANT:
# This is the embedding model used to create the missing
# 768-dimensional vectors in your Chroma collection.
EMBEDDING_MODEL = "all-mpnet-base-v2"

# Same retrieval size as the reference RAG-vs-WikiLLM repo
TOP_K = 30


# ============================================================
# LOAD MODELS
# ============================================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to Chroma...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = chroma_client.get_collection(
    name=COLLECTION_NAME
)

print(f"Chroma vectors: {collection.count()}")


# ============================================================
# AZURE CLIENT
# ============================================================

client = AzureOpenAI(
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT,
    api_key=AZURE_API_KEY,
)


# ============================================================
# RETRIEVE TOP-K CHUNKS
# ============================================================

def retrieve_chunks(question, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    return documents, metadatas, distances


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, documents):

    context_parts = []

    for i, document in enumerate(documents, start=1):
        context_parts.append(
            f"===== SOURCE CHUNK {i} =====\n"
            f"{document}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are answering questions about Rolls-Royce SMR engineering
and regulatory documentation.

Answer the question using ONLY the supplied context.

Do not use outside knowledge.

If the context does not contain enough information to answer,
say:

"Insufficient information in the retrieved documents."

Be precise and concise.

QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
"""

    response = client.chat.completions.create(
        model=AZURE_DEPLOYMENT,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_completion_tokens=1000
    )

    return response.choices[0].message.content


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if AZURE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        raise SystemExit(
            "ERROR: Put your Azure API key in AZURE_API_KEY first."
        )

    print("\n========================================")
    print("ROLLS-ROYCE SMR RAG")
    print("========================================")

    question = input("\nEnter your question:\n> ").strip()

    if not question:
        raise SystemExit("ERROR: Question cannot be empty.")

    print("\nRetrieving top 30 chunks...")

    documents, metadatas, distances = retrieve_chunks(
        question,
        TOP_K
    )

    print(f"Retrieved {len(documents)} chunks.")

    print("\nGenerating answer...")

    answer = generate_answer(
        question,
        documents
    )

    print("\n========================================")
    print("ANSWER")
    print("========================================")
    print(answer)

    print("\n========================================")
    print("RETRIEVED SOURCES")
    print("========================================")

    for i, metadata in enumerate(metadatas, start=1):

        source = metadata.get(
            "source_file",
            "unknown"
        )

        chunk = metadata.get(
            "chunk_index",
            "unknown"
        )

        print(
            f"{i:02d}. {source} | chunk {chunk}"
            f" | distance {distances[i - 1]:.4f}"
        )