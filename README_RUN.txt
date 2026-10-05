RAG vs WikiLLM evaluation pipeline

Run from:
C:\Users\Praveena\merlin-rag

1) Copy the four .py files into the project root.
2) Put your Azure API key into AZURE_API_KEY in each file.
3) Make sure your existing wiki/pages folder is complete.
4) Run:
   python generate_eval_dataset.py
5) Run:
   python build_wiki_index.py
6) Run:
   python run_comparison.py
7) Run:
   python evaluate_results.py

Outputs:
- evaluation/benchmark.json       frozen 50-question benchmark
- wiki_chroma_store/              Wiki concept index
- evaluation/raw_results.json    RAG + WikiLLM answers and token usage
- evaluation/final_results.json  per-question scores + summary
- evaluation/summary.csv         presentation-friendly summary

Important:
- Do NOT run prepare_rag.py again.
- Do NOT delete evaluation/benchmark.json once generated unless you intentionally want a new benchmark.
- The benchmark uses the same 3500/200 chunking as the current corpus.
- RAG uses top 30 chunks.
- WikiLLM uses top 30 index matches, keeps top 5 concepts, then performs one-hop expansion up to 10 related pages per concept.
- LLM reranking is intentionally omitted for speed; the reference repository uses an extra LLM reranking step. This version keeps the core retrieval/expansion design while avoiding 50 extra API calls.
