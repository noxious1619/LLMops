"""
Step 6: Retrieval Quality Validation Test Suite
Performs read-only Top-5 similarity search against the preserved Step 5 FAISS index
(faiss_index/step5_validation_test_20261101_121818) using real Google Gemini embeddings
(models/gemini-embedding-001).
Evaluates Hit Rate@5, MRR, Precision@5, and retrieval latency across 15 benchmark questions.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

# Reconfigure stdout for UTF-8 compatibility on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from langchain_community.vectorstores import FAISS
from multi_doc_chat.utils.model_loader import ModelLoader
from multi_doc_chat.logger import GLOBAL_LOGGER as log

INDEX_DIR = PROJECT_ROOT / "faiss_index" / "step5_validation_test_20261101_121818"
DATASET_PATH = PROJECT_ROOT / "tests" / "retrieval_evaluation_dataset.json"
RESULTS_PATH = PROJECT_ROOT / "tests" / "step6_retrieval_results.json"


def run_retrieval_validation():
    print("=" * 80)
    print("STEP 6: RAG RETRIEVAL QUALITY VALIDATION")
    print("=" * 80)

    # 1. Safety & Existence Check
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Target index directory does not exist: {INDEX_DIR}")
    faiss_file = INDEX_DIR / "index.faiss"
    pkl_file = INDEX_DIR / "index.pkl"
    if not faiss_file.exists() or not pkl_file.exists():
        raise FileNotFoundError(f"Index files missing in {INDEX_DIR}")

    print(f"Index directory: {INDEX_DIR}")
    print(f"FAISS index size: {faiss_file.stat().st_size:,} bytes")
    print(f"Docstore size:    {pkl_file.stat().st_size:,} bytes")

    # 2. Load Evaluation Dataset
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {DATASET_PATH}")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset: List[Dict[str, Any]] = json.load(f)
    print(f"Loaded evaluation questions: {len(dataset)} items from {DATASET_PATH.name}")

    # 3. Initialize ModelLoader & Embeddings
    print("Initializing ModelLoader for real Gemini embeddings...")
    loader = ModelLoader()
    embeddings = loader.load_embeddings()
    model_name = loader.config.get("embedding_model", {}).get("model_name", "unknown")
    print(f"Embedding model active: {model_name}")

    # 4. Load FAISS Index in Read-Only Mode
    print(f"Loading FAISS index from {INDEX_DIR}...")
    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings=embeddings,
        index_name="index",
        allow_dangerous_deserialization=True
    )
    total_vectors = vectorstore.index.ntotal
    dims = vectorstore.index.d
    print(f"FAISS loaded successfully. Vectors: {total_vectors}, Dimensions: {dims}")
    if total_vectors != 35 or dims != 3072:
        raise ValueError(f"Unexpected index specs: vectors={total_vectors} (expected 35), dims={dims} (expected 3072)")

    # 5. Execute Evaluation Queries
    print("\n" + "=" * 80)
    print("EXECUTING TOP-5 SIMILARITY SEARCH ACROSS EVALUATION DATASET")
    print("=" * 80)

    results = []
    latencies = []
    reciprocal_ranks = []
    chunk_hits = 0
    doc_hits = 0
    precision_at_5_scores = []

    for idx, item in enumerate(dataset, start=1):
        qid = item["question_id"]
        q_text = item["question"]
        expected_src = item["expected_source"]
        expected_cid = item["expected_chunk_id"]
        topic = item.get("topic", "N/A")

        print(f"\n--- [{idx:02d}/{len(dataset):02d}] {qid} ({topic}) ---")
        print(f"Query: {q_text}")
        print(f"Expected Source: {expected_src} | Expected Chunk ID: {expected_cid[:16]}...")

        # Time the similarity search with score
        t0 = time.perf_counter()
        search_results = vectorstore.similarity_search_with_score(q_text, k=5)
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        retrieved_items = []
        chunk_rank = None
        doc_rank = None
        relevant_chunks_in_top5 = 0

        for rank, (doc, score) in enumerate(search_results, start=1):
            cid = doc.metadata.get("chunk_id", "N/A")
            src_full = doc.metadata.get("source", "N/A")
            src_name = Path(src_full).name
            pg = doc.metadata.get("page")
            snippet = doc.page_content.replace("\n", " ")[:120]

            is_exact_chunk = (cid == expected_cid)
            is_same_doc = (src_name == expected_src)

            if is_exact_chunk and chunk_rank is None:
                chunk_rank = rank
            if is_same_doc and doc_rank is None:
                doc_rank = rank
            if is_exact_chunk or is_same_doc:
                relevant_chunks_in_top5 += 1

            retrieved_items.append({
                "rank": rank,
                "chunk_id": cid,
                "source": src_name,
                "page": pg,
                "score_l2": float(score),
                "is_target_chunk": is_exact_chunk,
                "is_target_doc": is_same_doc,
                "snippet": snippet
            })

            marker = "[TARGET CHUNK]" if is_exact_chunk else ("[SAME DOC]    " if is_same_doc else "              ")
            print(f"  Rank {rank} | L2: {score:.4f} | {marker} | {src_name} (p.{pg}) | ID: {cid[:12]}...")

        # Metrics for this query
        is_chunk_hit = (chunk_rank is not None)
        is_doc_hit = (doc_rank is not None)
        rr = (1.0 / chunk_rank) if is_chunk_hit else 0.0
        p5 = relevant_chunks_in_top5 / 5.0

        if is_chunk_hit:
            chunk_hits += 1
        if is_doc_hit:
            doc_hits += 1

        reciprocal_ranks.append(rr)
        precision_at_5_scores.append(p5)

        status_str = f"HIT at Rank {chunk_rank} (RR={rr:.4f})" if is_chunk_hit else "MISS"
        print(f"  Result: {status_str} | Doc Hit: {is_doc_hit} | Latency: {latency_ms:.1f}ms")

        results.append({
            "question_id": qid,
            "topic": topic,
            "question": q_text,
            "expected_source": expected_src,
            "expected_chunk_id": expected_cid,
            "latency_ms": latency_ms,
            "chunk_hit": is_chunk_hit,
            "chunk_rank": chunk_rank,
            "reciprocal_rank": rr,
            "doc_hit": is_doc_hit,
            "doc_rank": doc_rank,
            "precision_at_5": p5,
            "retrieved_chunks": retrieved_items
        })

    # 6. Aggregate Metrics Calculation
    total_q = len(dataset)
    hit_rate_chunk = (chunk_hits / total_q) * 100.0
    hit_rate_doc = (doc_hits / total_q) * 100.0
    mrr = sum(reciprocal_ranks) / total_q
    mean_p5 = sum(precision_at_5_scores) / total_q
    avg_latency = sum(latencies) / total_q
    min_latency = min(latencies)
    max_latency = max(latencies)
    sorted_latencies = sorted(latencies)
    median_latency = sorted_latencies[total_q // 2]

    # Rank Distribution
    rank_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "not_in_top5": 0}
    for r in results:
        cr = r["chunk_rank"]
        if cr in rank_distribution:
            rank_distribution[cr] += 1
        else:
            rank_distribution["not_in_top5"] += 1

    summary = {
        "index_evaluated": str(INDEX_DIR),
        "embedding_model": model_name,
        "embedding_dimensions": dims,
        "total_index_vectors": total_vectors,
        "total_questions_evaluated": total_q,
        "metrics": {
            "chunk_hit_rate_at_5_percent": round(hit_rate_chunk, 2),
            "doc_hit_rate_at_5_percent": round(hit_rate_doc, 2),
            "mean_reciprocal_rank_mrr": round(mrr, 4),
            "mean_precision_at_5": round(mean_p5, 4),
            "latency_ms": {
                "avg": round(avg_latency, 2),
                "min": round(min_latency, 2),
                "max": round(max_latency, 2),
                "median": round(median_latency, 2)
            },
            "rank_distribution": rank_distribution
        },
        "query_results": results
    }

    # Save to JSON
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDetailed evaluation results saved to: {RESULTS_PATH}")

    # 7. Print Final Table
    print("\n" + "=" * 80)
    print("STEP 6: RETRIEVAL QUALITY VALIDATION SUMMARY REPORT")
    print("=" * 80)
    print(f"Total Questions Evaluated:  {total_q}")
    print(f"Exact Chunk Hit Rate @ 5:   {hit_rate_chunk:.2f}% ({chunk_hits}/{total_q})")
    print(f"Document Hit Rate @ 5:      {hit_rate_doc:.2f}% ({doc_hits}/{total_q})")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Mean Precision @ 5:         {mean_p5:.4f}")
    print(f"Average Retrieval Latency:  {avg_latency:.1f} ms (Min: {min_latency:.1f}ms, Max: {max_latency:.1f}ms, Median: {median_latency:.1f}ms)")
    print("-" * 80)
    print(f"Rank Distribution: Rank 1: {rank_distribution[1]} | Rank 2: {rank_distribution[2]} | Rank 3: {rank_distribution[3]} | Rank 4: {rank_distribution[4]} | Rank 5: {rank_distribution[5]} | Miss: {rank_distribution['not_in_top5']}")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    run_retrieval_validation()
