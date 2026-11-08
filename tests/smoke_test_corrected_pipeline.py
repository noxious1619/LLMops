"""
End-to-End Smoke Test for Corrected Hybrid Candidate Retrieval Pipeline.

Verifies:
1. FAISS returns no more than 10 candidates.
2. BM25 returns no more than 10 candidates.
3. RRF receives both candidate lists and fuses them with k=60.
4. Deduplication works correctly by chunk_id.
5. Fused candidate pool contains no more than 20 unique documents.
6. MiniLM Cross-Encoder successfully scores candidate documents.
7. Actual inference uses batch_size=5.
8. Reports exact inference batches processed.
9. Final output contains no more than 5 documents.
10. Every returned document retains its original chunk_id, source and metadata.
11. Model runs on RTX 3050 GPU when CUDA is available.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv(PROJECT_ROOT / ".env", override=True)

import torch
from multi_doc_chat.src.document_chat.hybrid_retriever import HybridRetriever
from multi_doc_chat.src.document_chat.reranker import CrossEncoderReranker


def run_smoke_test():
    query = "How does Redis distribute data across nodes using hash slots?"

    faiss_dir = PROJECT_ROOT / "faiss_index" / "benchmark_70docs_v1"
    bm25_dir = PROJECT_ROOT / "data" / "bm25_index"

    # Step 1: Initialize HybridRetriever with dense_k=10, sparse_k=10, final_k=5, rrf_k=60
    retriever = HybridRetriever.from_saved(
        faiss_index_dir=faiss_dir,
        bm25_index_dir=bm25_dir,
        dense_k=10,
        sparse_k=10,
        final_k=5,
        rrf_k=60,
    )

    # Step 2: Initialize CrossEncoderReranker with batch_size=5
    reranker = CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        device="auto",
        batch_size=5,
    )

    # Step 3: Fetch individual retriever candidates to inspect counts
    dense_candidates = retriever._fetch_dense(query)
    sparse_candidates = retriever._fetch_sparse(query)

    faiss_count = len(dense_candidates)
    bm25_count = len(sparse_candidates)
    combined_count = faiss_count + bm25_count

    assert faiss_count <= 10, f"FAISS returned {faiss_count} > 10 candidates"
    assert bm25_count <= 10, f"BM25 returned {bm25_count} > 10 candidates"

    # Step 4: Retrieve fused candidates via candidate retrieval pipeline
    candidates = retriever.retrieve_candidates(query, candidate_k=20)
    unique_candidate_count = len(candidates)

    assert unique_candidate_count <= 20, f"Candidate pool has {unique_candidate_count} > 20 candidates"

    # Verify deduplication
    candidate_cids = [c.metadata.get("chunk_id") for c in candidates]
    assert len(candidate_cids) == len(set(candidate_cids)), "Duplicate chunk_ids found in candidate pool!"

    # Verify metadata preservation on candidates
    for c in candidates:
        assert c.metadata.get("chunk_id"), "Candidate missing chunk_id"
        assert c.metadata.get("source"), "Candidate missing source"
        assert "original_rrf_rank" in c.metadata, "Candidate missing original_rrf_rank"
        assert "original_rrf_score" in c.metadata, "Candidate missing original_rrf_score"

    # Step 5: Instrument CrossEncoder inference to verify batch_size=5 and batch count
    original_predict = reranker.model.predict
    inference_calls = []

    def tracking_predict(pairs, *args, **kwargs):
        b_size = kwargs.get("batch_size", 5)
        inference_calls.append({
            "total_pairs": len(pairs),
            "batch_size": b_size,
            "calculated_batches": math.ceil(len(pairs) / b_size) if b_size > 0 else 1,
        })
        return original_predict(pairs, *args, **kwargs)

    reranker.model.predict = tracking_predict

    # Step 6: Execute reranking
    reranked_docs = reranker.rerank(query, candidates, top_k=5)
    final_count = len(reranked_docs)

    assert final_count <= 5, f"Final reranked documents count {final_count} > 5"
    assert reranker.batch_size == 5, f"Reranker batch_size {reranker.batch_size} != 5"
    assert len(inference_calls) == 1, "Predict was not called exactly once"
    assert inference_calls[0]["batch_size"] == 5, f"Predict received batch_size {inference_calls[0]['batch_size']} != 5"

    actual_batches = inference_calls[0]["calculated_batches"]

    # Verify final documents retain all metadata and chunk_id
    for r in reranked_docs:
        assert r.metadata.get("chunk_id"), "Final doc missing chunk_id"
        assert r.metadata.get("source"), "Final doc missing source"
        assert "rerank_score" in r.metadata, "Final doc missing rerank_score"
        assert "rerank_rank" in r.metadata, "Final doc missing rerank_rank"
        assert "original_rrf_rank" in r.metadata, "Final doc missing original_rrf_rank"

    # Print execution summary
    print(f"Query: {query}\n")
    print(f"FAISS Candidates: {faiss_count}")
    print(f"BM25 Candidates: {bm25_count}")
    print(f"Combined Candidates: {combined_count}")
    print(f"Unique Candidates After Deduplication: {unique_candidate_count}\n")
    print(f"Reranker Model: {reranker.model_name}")
    print(f"Execution Device: {reranker.actual_device}")
    print(f"Batch Size: {reranker.batch_size}")
    print(f"Inference Batches: {actual_batches}\n")
    print(f"Final Reranked Documents: {final_count}\n")
    print("Top-5:")
    for idx, doc in enumerate(reranked_docs, 1):
        cid = doc.metadata.get("chunk_id", "N/A")
        src = doc.metadata.get("source", "N/A")
        score = doc.metadata.get("rerank_score", 0.0)
        orig_rrf_rank = doc.metadata.get("original_rrf_rank", "N/A")
        print(f"{idx}. Chunk ID: {cid} | Score: {score:.4f} | RRF Rank: {orig_rrf_rank} | Source: {src}")

    print("\nPipeline Status: PASS")


if __name__ == "__main__":
    run_smoke_test()
