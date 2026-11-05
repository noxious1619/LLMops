import json
from pathlib import Path

with open("tests/step8_50_retrieval_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

# Also load docstore chunk text to see snippets of what was retrieved
import pickle
with open("faiss_index/benchmark_70docs_v1/index.pkl", "rb") as f:
    docstore, index_to_docstore_id = pickle.load(f)

chunk_content = {}
for idx, doc_id in index_to_docstore_id.items():
    doc = docstore.search(doc_id)
    cid = doc.metadata.get("chunk_id", doc_id)
    chunk_content[cid] = doc.page_content

non_r1 = [r for r in results if r["first_relevant_rank"] != 1]
print(f"Total non-Rank-1 cases: {len(non_r1)}")

for r in non_r1:
    print("=" * 80)
    print(f"Question ID: {r['question_id']}")
    print(f"Question:    {r['question']}")
    print(f"Category:    {r['question_type']} | Difficulty: {r['difficulty']}")
    print(f"Expected Target Document: {r['source_documents'][0]}")
    print(f"First Relevant Hit Rank:   Rank {r['first_relevant_rank']}")
    print("-" * 80)
    print("Top 5 Retrieved Chunks:")
    for i, (cid, src, score) in enumerate(zip(r["retrieved_chunk_ids"], r["retrieved_sources"], r["retrieved_scores"]), 1):
        is_target = cid in r["relevant_chunk_ids"]
        tag = "🎯 [TARGET CHUNK]" if is_target else "⚠️ [COMPETING CHUNK]"
        snippet = chunk_content.get(cid, "")[:140].replace("\n", " ")
        print(f"  Pos {i} (Score: {score:.4f}) {tag} -> {src}")
        print(f"        Snippet: {snippet}...")
    print()

