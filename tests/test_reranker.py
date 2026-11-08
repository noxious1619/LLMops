"""
Unit tests for the CrossEncoderReranker module.
Tests device detection, GPU/CPU initialization, sorting, metadata preservation,
tie-breaking, edge cases, acronym handling, and OOM recovery.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
from langchain_core.documents import Document

from multi_doc_chat.src.document_chat.reranker import CrossEncoderReranker


class TestCrossEncoderRerankerUnit(unittest.TestCase):
    """Unit test suite for CrossEncoderReranker."""

    def setUp(self):
        self.sample_docs = [
            Document(
                page_content="BGP is a path-vector routing protocol used between autonomous systems on the Internet.",
                metadata={"chunk_id": "chunk_001", "source": "network_guide.pdf", "original_rrf_rank": 1, "original_rrf_score": 0.032},
            ),
            Document(
                page_content="OAuth 2.0 is an authorization framework allowing third-party access to HTTP services.",
                metadata={"chunk_id": "chunk_002", "source": "auth_spec.docx", "original_rrf_rank": 2, "original_rrf_score": 0.030},
            ),
            Document(
                page_content="Redis is an in-memory key-value data store used as a database, cache, and message broker.",
                metadata={"chunk_id": "chunk_003", "source": "redis_doc.txt", "original_rrf_rank": 3, "original_rrf_score": 0.028},
            ),
        ]

    def test_01_reranker_init_gpu(self):
        """Test explicit GPU initialization if CUDA is available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available on this environment.")

        reranker = CrossEncoderReranker(device="cuda:0")
        self.assertIn("cuda", reranker.actual_device)
        self.assertIsNotNone(reranker.model)

    def test_02_auto_gpu_detection(self):
        """Test device='auto' selects cuda:0 when CUDA is available."""
        if torch.cuda.is_available():
            reranker = CrossEncoderReranker(device="auto")
            self.assertEqual(reranker.actual_device, "cuda:0")
        else:
            reranker = CrossEncoderReranker(device="auto")
            self.assertEqual(reranker.actual_device, "cpu")

    def test_03_explicit_cpu_override(self):
        """Test explicit CPU override selects cpu."""
        reranker = CrossEncoderReranker(device="cpu")
        self.assertEqual(reranker.actual_device, "cpu")

    def test_04_fallback_from_cuda_to_cpu(self):
        """Test fallback from CUDA to CPU when CUDA is unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            reranker = CrossEncoderReranker(device="cuda:0", allow_cpu_fallback=True)
            self.assertEqual(reranker.actual_device, "cpu")

    def test_05_rescoring_changes_ranking_order(self):
        """Test that reranking rescores documents and changes the ranking order."""
        reranker = CrossEncoderReranker(device="cpu")
        # Invert the order via mocked predict: chunk_003 gets highest score, chunk_001 gets lowest
        with patch.object(reranker.model, "predict", return_value=[1.2, 5.8, 9.4]):
            reranked = reranker.rerank("Tell me about Redis", self.sample_docs, top_k=3)

        self.assertEqual(len(reranked), 3)
        self.assertEqual(reranked[0].metadata["chunk_id"], "chunk_003")
        self.assertEqual(reranked[1].metadata["chunk_id"], "chunk_002")
        self.assertEqual(reranked[2].metadata["chunk_id"], "chunk_001")

    def test_06_preserves_all_original_metadata(self):
        """Test that original document metadata fields are fully preserved."""
        reranker = CrossEncoderReranker(device="cpu")
        with patch.object(reranker.model, "predict", return_value=[5.0, 4.0, 3.0]):
            reranked = reranker.rerank("Query", self.sample_docs, top_k=3)

        for doc in reranked:
            self.assertIn("chunk_id", doc.metadata)
            self.assertIn("source", doc.metadata)

        self.assertEqual(reranked[0].metadata["source"], "network_guide.pdf")

    def test_07_adds_rerank_metadata(self):
        """Test that rerank_score, rerank_rank, and original_rrf_rank are present."""
        reranker = CrossEncoderReranker(device="cpu")
        with patch.object(reranker.model, "predict", return_value=[7.5, 2.1, 8.9]):
            reranked = reranker.rerank("Query", self.sample_docs, top_k=3)

        # Expected order: chunk_003 (8.9), chunk_001 (7.5), chunk_002 (2.1)
        self.assertEqual(reranked[0].metadata["chunk_id"], "chunk_003")
        self.assertEqual(reranked[0].metadata["rerank_rank"], 1)
        self.assertAlmostEqual(reranked[0].metadata["rerank_score"], 8.9, places=2)
        self.assertEqual(reranked[0].metadata["original_rrf_rank"], 3)

    def test_08_limits_output_to_top_k(self):
        """Test that output length strictly obeys top_k."""
        reranker = CrossEncoderReranker(device="cpu")
        with patch.object(reranker.model, "predict", return_value=[3.0, 2.0, 1.0]):
            top_2 = reranker.rerank("Query", self.sample_docs, top_k=2)
            top_1 = reranker.rerank("Query", self.sample_docs, top_k=1)

        self.assertEqual(len(top_2), 2)
        self.assertEqual(len(top_1), 1)

    def test_09_empty_input_returns_empty(self):
        """Test edge cases with empty input, empty query, and top_k <= 0."""
        reranker = CrossEncoderReranker(device="cpu")
        self.assertEqual(reranker.rerank("Query", []), [])
        self.assertEqual(reranker.rerank("", self.sample_docs), [])
        self.assertEqual(reranker.rerank("   ", self.sample_docs), [])
        self.assertEqual(reranker.rerank("Query", self.sample_docs, top_k=0), [])
        self.assertEqual(reranker.rerank("Query", self.sample_docs, top_k=-5), [])

    def test_10_single_document_input(self):
        """Test single document input assigns score and rank=1 without sorting."""
        reranker = CrossEncoderReranker(device="cpu")
        single = [self.sample_docs[0]]
        with patch.object(reranker.model, "predict", return_value=[6.4]):
            results = reranker.rerank("Query", single, top_k=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].metadata["chunk_id"], "chunk_001")
        self.assertEqual(results[0].metadata["rerank_rank"], 1)
        self.assertAlmostEqual(results[0].metadata["rerank_score"], 6.4, places=2)

    def test_11_duplicate_chunk_ids_deduplicated(self):
        """Test that duplicate chunk_ids in input are deduplicated (first kept)."""
        reranker = CrossEncoderReranker(device="cpu")
        dupe_doc = Document(
            page_content="Duplicate BGP chunk",
            metadata={"chunk_id": "chunk_001", "source": "network_guide.pdf", "original_rrf_rank": 4},
        )
        docs_with_dupe = [self.sample_docs[0], dupe_doc, self.sample_docs[1]]

        with patch.object(reranker.model, "predict", return_value=[5.0, 4.0]):
            results = reranker.rerank("Query", docs_with_dupe, top_k=5)

        self.assertEqual(len(results), 2)
        chunk_ids = [d.metadata["chunk_id"] for d in results]
        self.assertEqual(chunk_ids, ["chunk_001", "chunk_002"])
        # Verify first occurrence's content was preserved
        self.assertEqual(results[0].page_content, self.sample_docs[0].page_content)

    def test_12_deterministic_tie_breaking(self):
        """Test that ties in score are broken by original RRF rank, then chunk_id."""
        reranker = CrossEncoderReranker(device="cpu")
        tied_docs = [
            Document(
                page_content="Doc B",
                metadata={"chunk_id": "chunk_b", "original_rrf_rank": 3},
            ),
            Document(
                page_content="Doc A",
                metadata={"chunk_id": "chunk_a", "original_rrf_rank": 1},
            ),
            Document(
                page_content="Doc C",
                metadata={"chunk_id": "chunk_c", "original_rrf_rank": 2},
            ),
        ]

        # All docs receive identical score 5.0
        with patch.object(reranker.model, "predict", return_value=[5.0, 5.0, 5.0]):
            results = reranker.rerank("Query", tied_docs, top_k=3)

        # Expected order sorted by original_rrf_rank: 1 (Doc A), 2 (Doc C), 3 (Doc B)
        self.assertEqual([d.metadata["chunk_id"] for d in results], ["chunk_a", "chunk_c", "chunk_b"])

        # Secondary tie-break: if original_rrf_rank is also equal, sort by chunk_id
        tied_rrf = [
            Document(page_content="Doc Z", metadata={"chunk_id": "chunk_z", "original_rrf_rank": 1}),
            Document(page_content="Doc M", metadata={"chunk_id": "chunk_m", "original_rrf_rank": 1}),
        ]
        with patch.object(reranker.model, "predict", return_value=[4.0, 4.0]):
            results_rrf = reranker.rerank("Query", tied_rrf, top_k=2)

        self.assertEqual([d.metadata["chunk_id"] for d in results_rrf], ["chunk_m", "chunk_z"])

    def test_13_long_text_truncation_handled(self):
        """Test that inputs exceeding 512 tokens are truncated without crashing."""
        reranker = CrossEncoderReranker(device="cpu")
        long_content = "Word " * 1200  # ~1200 words / tokens
        long_doc = Document(
            page_content=long_content,
            metadata={"chunk_id": "chunk_long", "original_rrf_rank": 1},
        )
        # Real inference call with long text to verify tokenizer truncation
        results = reranker.rerank("Test query", [long_doc], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("rerank_score", results[0].metadata)

    def test_14_technical_acronyms_relevance(self):
        """Test real cross-encoder scores higher for matching technical acronym."""
        reranker = CrossEncoderReranker(device="cpu")
        bgp_doc = Document(
            page_content="The Border Gateway Protocol (BGP) manages how packets are routed across the Internet.",
            metadata={"chunk_id": "bgp_doc", "original_rrf_rank": 2},
        )
        cooking_doc = Document(
            page_content="A classic Italian risotto requires arborio rice and chicken broth.",
            metadata={"chunk_id": "cooking_doc", "original_rrf_rank": 1},
        )
        results = reranker.rerank("How does BGP routing work?", [cooking_doc, bgp_doc], top_k=2)
        # BGP doc must be ranked first by the real cross-encoder
        self.assertEqual(results[0].metadata["chunk_id"], "bgp_doc")
        self.assertGreater(results[0].metadata["rerank_score"], results[1].metadata["rerank_score"])

    def test_15_batch_size_parameter_respected(self):
        """Test default batch_size is 5 and passed to model.predict."""
        reranker_default = CrossEncoderReranker(device="cpu")
        self.assertEqual(reranker_default.batch_size, 5)
        with patch.object(reranker_default.model, "predict", return_value=[1.0, 2.0, 3.0]) as mock_pred:
            reranker_default.rerank("Query", self.sample_docs, top_k=3)
            mock_pred.assert_called_once()
            _, kwargs = mock_pred.call_args
            self.assertEqual(kwargs.get("batch_size"), 5)

        reranker_custom = CrossEncoderReranker(device="cpu", batch_size=7)
        self.assertEqual(reranker_custom.batch_size, 7)
        with patch.object(reranker_custom.model, "predict", return_value=[1.0, 2.0, 3.0]) as mock_pred:
            reranker_custom.rerank("Query", self.sample_docs, top_k=3)
            mock_pred.assert_called_once()
            _, kwargs = mock_pred.call_args
            self.assertEqual(kwargs.get("batch_size"), 7)

    def test_16_cuda_oom_recovery(self):
        """Test CUDA OOM recovery: clears cache and falls back to CPU when allowed."""
        reranker = CrossEncoderReranker(device="cpu")
        # Simulate that actual_device was cuda:0
        reranker.actual_device = "cuda:0"
        reranker.allow_cpu_fallback = True

        call_count = 0

        def mock_predict(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call simulates CUDA OOM
                raise torch.cuda.OutOfMemoryError("CUDA out of memory in test")
            return [9.0, 8.0, 7.0]

        with patch.object(reranker.model, "predict", side_effect=mock_predict), \
             patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.empty_cache") as mock_empty_cache:
            results = reranker.rerank("Query", self.sample_docs, top_k=3)

        mock_empty_cache.assert_called_once()
        self.assertEqual(reranker.actual_device, "cpu")
        self.assertEqual(len(results), 3)

    def test_17_model_loading_failure_raises_descriptive_error(self):
        """Test that invalid model loading without fallback raises descriptive RuntimeError."""
        with self.assertRaises(RuntimeError) as ctx:
            CrossEncoderReranker(
                model_name="nonexistent/model-that-does-not-exist-xyz123",
                device="cpu",
                allow_cpu_fallback=False,
            )
        self.assertIn("Unable to load CrossEncoder model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

