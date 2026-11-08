"""
Integration tests for Cross-Encoder Reranking with HybridRetriever.
Validates the end-to-end candidate retrieval + reranking pipeline,
GPU VRAM stability, latency benchmarks, and backward compatibility.
"""

from __future__ import annotations

import json
import logging
import os
import time
import unittest
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

import torch
from langchain_core.documents import Document

from multi_doc_chat.src.document_chat.hybrid_retriever import HybridRetriever
from multi_doc_chat.src.document_chat.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

FAISS_DIR = Path("faiss_index/benchmark_70docs_v1")
BM25_DIR = Path("data/bm25_index")
BENCHMARK_PATH = Path("tests/retrieval_ground_truth_200_corrected_v1.json")


class TestRerankerIntegration(unittest.TestCase):
    """Integration test suite executing real index queries with CrossEncoderReranker."""

    @classmethod
    def setUpClass(cls):
        """Set up HybridRetriever and CrossEncoderReranker once for integration tests."""
        if not FAISS_DIR.exists():
            raise FileNotFoundError(f"FAISS directory {FAISS_DIR} not found.")

        cls.retriever = HybridRetriever.from_saved(
            faiss_index_dir=FAISS_DIR,
            bm25_index_dir=BM25_DIR,
            dense_k=10,
            sparse_k=10,
            final_k=5,
            rrf_k=60,
        )
        cls.reranker = CrossEncoderReranker(device="auto", batch_size=5)

        cls.benchmark_questions = []
        if BENCHMARK_PATH.exists():
            with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                cls.benchmark_questions = [item["question"] for item in data[:5]]
        else:
            cls.benchmark_questions = [
                "What is the fundamental difference between authentication and authorization?",
                "How does the Proof Key for Code Exchange (PKCE) extension protect authorization flows?",
                "What is Border Gateway Protocol BGP and how do autonomous systems route traffic?",
                "Explain Redis in-memory caching and eviction policies.",
                "How does JWT token signing and verification operate cryptographically?",
            ]

    def test_01_end_to_end_candidate_and_rerank(self):
        """Test full candidate retrieval (k=20) through rerank (top_k=5) pipeline."""
        query = self.benchmark_questions[0]
        candidates = self.retriever.retrieve_candidates(query, candidate_k=20)

        self.assertGreater(len(candidates), 0)
        self.assertLessEqual(len(candidates), 20)

        # Verify candidate metadata annotations
        for c in candidates:
            self.assertIn("chunk_id", c.metadata)
            self.assertIn("original_rrf_rank", c.metadata)
            self.assertIn("original_rrf_score", c.metadata)

        reranked = self.reranker.rerank(query, candidates, top_k=5)
        self.assertEqual(len(reranked), min(5, len(candidates)))

        for r in reranked:
            self.assertIsInstance(r, Document)
            self.assertIn("rerank_score", r.metadata)
            self.assertIn("rerank_rank", r.metadata)

    def test_02_output_contains_exactly_5_with_valid_scores(self):
        """Test that reranked output has exactly 5 docs with descending rerank_scores."""
        query = self.benchmark_questions[1]
        candidates = self.retriever.retrieve_candidates(query, candidate_k=20)
        reranked = self.reranker.rerank(query, candidates, top_k=5)

        self.assertEqual(len(reranked), 5)
        ranks = [doc.metadata["rerank_rank"] for doc in reranked]
        self.assertEqual(ranks, [1, 2, 3, 4, 5])

        # Verify descending order of scores
        scores = [doc.metadata["rerank_score"] for doc in reranked]
        for i in range(len(scores) - 1):
            self.assertGreaterEqual(
                scores[i],
                scores[i + 1],
                f"Score at rank {i+1} ({scores[i]}) is less than score at rank {i+2} ({scores[i+1]})",
            )

    def test_03_pipeline_latency_measurement(self):
        """Test and measure candidate retrieval latency + rerank latency."""
        query = self.benchmark_questions[2]

        # Warmup pass
        c_warm = self.retriever.retrieve_candidates(query, candidate_k=20)
        _ = self.reranker.rerank(query, c_warm, top_k=5)

        retrieval_latencies = []
        rerank_latencies = []
        total_latencies = []

        for q in self.benchmark_questions:
            t0 = time.perf_counter()
            candidates = self.retriever.retrieve_candidates(q, candidate_k=20)
            t1 = time.perf_counter()
            reranked = self.reranker.rerank(q, candidates, top_k=5)
            t2 = time.perf_counter()

            retrieval_ms = (t1 - t0) * 1000
            rerank_ms = (t2 - t1) * 1000
            total_ms = (t2 - t0) * 1000

            retrieval_latencies.append(retrieval_ms)
            rerank_latencies.append(rerank_ms)
            total_latencies.append(total_ms)

        avg_retrieval = sum(retrieval_latencies) / len(retrieval_latencies)
        avg_rerank = sum(rerank_latencies) / len(rerank_latencies)
        avg_total = sum(total_latencies) / len(total_latencies)

        logger.info(
            f"Latency Benchmark (5 queries): "
            f"Candidate Retrieval: {avg_retrieval:.2f} ms | "
            f"Reranking: {avg_rerank:.2f} ms | "
            f"Total Pipeline: {avg_total:.2f} ms"
        )

        # On GPU, reranker for 20 pairs should comfortably execute in < 200 ms
        if torch.cuda.is_available() and "cuda" in self.reranker.actual_device:
            self.assertLess(avg_rerank, 200.0, f"Rerank latency on GPU too high: {avg_rerank:.2f} ms")

    def test_04_gpu_memory_management_no_leak(self):
        """Test that running multiple rerank passes does not leak VRAM."""
        if not torch.cuda.is_available() or "cuda" not in self.reranker.actual_device:
            self.skipTest("CUDA GPU not active.")

        query = self.benchmark_questions[3]
        candidates = self.retriever.retrieve_candidates(query, candidate_k=20)

        # Warmup pass
        _ = self.reranker.rerank(query, candidates, top_k=5)
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated(0)

        # Execute 10 consecutive rerank operations
        for _ in range(10):
            _ = self.reranker.rerank(query, candidates, top_k=5)

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(0)

        delta_mb = (mem_after - mem_before) / (1024 ** 2)
        # Delta should be near 0 MB (no memory leak)
        self.assertLess(
            delta_mb,
            5.0,
            f"VRAM leak detected: allocated memory grew by {delta_mb:.2f} MB across 10 passes.",
        )

    def test_05_generation_pipeline_compatibility(self):
        """Test output documents format seamlessly into standard RAG generation prompts."""
        query = self.benchmark_questions[4]
        candidates = self.retriever.retrieve_candidates(query, candidate_k=20)
        reranked = self.reranker.rerank(query, candidates, top_k=5)

        # Simulate context assembly in ConversationalRAG
        context_blocks = []
        for doc in reranked:
            source = doc.metadata.get("source", "unknown")
            chunk_id = doc.metadata.get("chunk_id", "unknown")
            block = f"[Source: {source} | Chunk: {chunk_id}]\n{doc.page_content}"
            context_blocks.append(block)

        formatted_context = "\n\n---\n\n".join(context_blocks)
        self.assertIsInstance(formatted_context, str)
        self.assertGreater(len(formatted_context), 100)

    def test_06_compare_reranked_vs_hybrid_only(self):
        """Compare hybrid-only Top-5 vs reranked Top-5 on 5 sample queries."""
        comparisons = []
        for idx, q in enumerate(self.benchmark_questions, start=1):
            hybrid_top5 = self.retriever.retrieve(q)
            hybrid_cids = [d.metadata.get("chunk_id") for d in hybrid_top5]

            candidates = self.retriever.retrieve_candidates(q, candidate_k=20)
            reranked_top5 = self.reranker.rerank(q, candidates, top_k=5)
            reranked_cids = [d.metadata.get("chunk_id") for d in reranked_top5]

            rank_1_match = hybrid_cids[0] == reranked_cids[0]
            overlap_count = len(set(hybrid_cids).intersection(set(reranked_cids)))

            comparisons.append({
                "query_idx": idx,
                "query": q,
                "hybrid_top_chunk": hybrid_cids[0],
                "reranked_top_chunk": reranked_cids[0],
                "rank_1_match": rank_1_match,
                "overlap_in_top5": overlap_count,
                "hybrid_top5_cids": hybrid_cids,
                "reranked_top5_cids": reranked_cids,
            })

        self.assertEqual(len(comparisons), 5)
        logger.info(f"Reranking vs Hybrid comparisons completed for {len(comparisons)} queries.")


if __name__ == "__main__":
    unittest.main()
