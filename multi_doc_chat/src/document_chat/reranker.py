"""
Cross-Encoder Reranker Module for Two-Stage Hybrid Retrieval.

This module provides a production-grade Cross-Encoder reranker using
`cross-encoder/ms-marco-MiniLM-L-6-v2` with GPU acceleration (NVIDIA RTX 3050),
automatic device detection, graceful CUDA OOM recovery, and deterministic ranking.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import uuid

import torch
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Production-grade Cross-Encoder Reranker for re-scoring retrieved candidate chunks.

    Attributes:
        model_name: Name or path of the sentence-transformers Cross-Encoder model.
        device: Target execution device ('auto', 'cuda:0', 'cuda', or 'cpu').
        batch_size: Batch size for cross-encoder inference (default: 5).
        allow_cpu_fallback: Whether to fall back to CPU if CUDA is unavailable or encounters OOM.
        max_length: Maximum sequence length for tokenization (default: 512).
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str = "auto",
        batch_size: int = 5,
        allow_cpu_fallback: bool = True,
        max_length: int = 512,
    ):
        """
        Initialize the CrossEncoderReranker with GPU-first loading logic and fallback mechanisms.

        Args:
            model_name: HuggingFace model hub ID or local directory path.
            device: Requested device ('auto', 'cuda:0', 'cuda', 'cpu').
            batch_size: Inference batch size (default: 5).
            allow_cpu_fallback: Allow graceful CPU fallback on CUDA failure / OOM.
            max_length: Maximum concatenated token length for the cross-encoder.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")
        if max_length <= 0:
            raise ValueError(f"max_length must be a positive integer, got {max_length}")

        self.model_name = model_name
        self.device_request = device
        self.batch_size = batch_size
        self.allow_cpu_fallback = allow_cpu_fallback
        self.max_length = max_length

        self.actual_device: str = self._resolve_target_device(device)
        self.model: CrossEncoder = self._load_model(self.actual_device)

    def _resolve_target_device(self, device_str: str) -> str:
        """Resolve requested device string to concrete device name."""
        clean_dev = device_str.strip().lower()
        if clean_dev == "auto":
            if torch.cuda.is_available():
                return "cuda:0"
            logger.info("CUDA is not available. Auto-selected device: 'cpu'.")
            return "cpu"

        if clean_dev.startswith("cuda"):
            if not torch.cuda.is_available():
                if self.allow_cpu_fallback:
                    logger.warning(
                        f"Device '{device_str}' requested, but CUDA is unavailable. "
                        "allow_cpu_fallback=True: Falling back to 'cpu'."
                    )
                    return "cpu"
                raise RuntimeError(
                    f"CUDA device '{device_str}' requested but CUDA is not available on this system."
                )
            if clean_dev == "cuda":
                return "cuda:0"
            return clean_dev

        return clean_dev

    def _load_model(self, target_device: str) -> CrossEncoder:
        """Load the CrossEncoder model onto target device with fallback handling."""
        logger.info(f"Loading CrossEncoder model '{self.model_name}' on device '{target_device}'...")

        try:
            model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=target_device,
            )
            self._log_device_info(target_device)
            return model

        except Exception as err:
            logger.error(
                f"Failed to load model '{self.model_name}' on device '{target_device}': {err}",
                exc_info=True,
            )
            if "cuda" in target_device and self.allow_cpu_fallback:
                logger.warning("Attempting emergency fallback: loading CrossEncoder on CPU...")
                try:
                    fallback_model = CrossEncoder(
                        self.model_name,
                        max_length=self.max_length,
                        device="cpu",
                    )
                    self.actual_device = "cpu"
                    logger.info("Successfully loaded CrossEncoder on CPU fallback.")
                    return fallback_model
                except Exception as fallback_err:
                    raise RuntimeError(
                        f"CPU fallback also failed for '{self.model_name}': {fallback_err}"
                    ) from fallback_err
            raise RuntimeError(
                f"Unable to load CrossEncoder model '{self.model_name}' on '{target_device}': {err}"
            ) from err

    def _log_device_info(self, device: str) -> None:
        """Log diagnostic hardware details for the active device."""
        if "cuda" in device and torch.cuda.is_available():
            dev_idx = 0
            if ":" in device:
                try:
                    dev_idx = int(device.split(":")[1])
                except ValueError:
                    dev_idx = 0
            gpu_name = torch.cuda.get_device_name(dev_idx)
            props = torch.cuda.get_device_properties(dev_idx)
            total_vram_mb = props.total_memory / (1024 ** 2)
            alloc_vram_mb = torch.cuda.memory_allocated(dev_idx) / (1024 ** 2)
            res_vram_mb = torch.cuda.memory_reserved(dev_idx) / (1024 ** 2)
            logger.info(
                f"CrossEncoder active on GPU [{dev_idx}]: {gpu_name} | "
                f"Allocated VRAM: {alloc_vram_mb:.2f} MB | "
                f"Reserved VRAM: {res_vram_mb:.2f} MB | "
                f"Total VRAM: {total_vram_mb:.2f} MB"
            )
        else:
            logger.info(f"CrossEncoder active on device: {device}")

    def _predict_with_guard(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Execute prediction with CUDA OOM detection, memory cleanup, and CPU fallback.
        """
        if not pairs:
            return []

        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            # Ensure float conversion
            if hasattr(raw_scores, "tolist"):
                return [float(s) for s in raw_scores.tolist()]
            return [float(s) for s in raw_scores]

        except Exception as exc:
            is_cuda_oom = (
                isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", RuntimeError))
                or "out of memory" in str(exc).lower()
                or "cuda error" in str(exc).lower()
            )

            if is_cuda_oom:
                logger.error(
                    f"CUDA OutOfMemory / CUDA error encountered during CrossEncoder inference: {exc}"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if self.allow_cpu_fallback and "cuda" in self.actual_device:
                    logger.warning("Executing emergency CPU fallback for inference batch...")
                    try:
                        if hasattr(self.model, "model") and hasattr(self.model.model, "to"):
                            self.model.model.to("cpu")
                        self.actual_device = "cpu"

                        fallback_scores = self.model.predict(
                            pairs,
                            batch_size=self.batch_size,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                        )
                        if hasattr(fallback_scores, "tolist"):
                            return [float(s) for s in fallback_scores.tolist()]
                        return [float(s) for s in fallback_scores]
                    except Exception as cpu_exc:
                        logger.error(f"CPU fallback inference failed: {cpu_exc}")
                        raise cpu_exc from exc

            raise exc

    def rerank_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Tuple[Document, float, int]]:
        """
        Rerank a list of candidate documents against a query string.

        Args:
            query: The search query string.
            documents: Candidate documents retrieved by first-stage retrieval.
            top_k: Number of reranked documents to return (default: 5).

        Returns:
            List of tuples: (reranked_document, score, rank).
        """
        if not documents or top_k <= 0:
            return []

        if not query or not query.strip():
            logger.warning("Empty or whitespace query provided to rerank_with_scores; returning empty list.")
            return []

        clean_query = query.strip()

        # Step 1: Enforce chunk_id deduplication (preserve first occurrence)
        unique_docs: List[Document] = []
        seen_chunk_ids: set[str] = set()

        for idx, doc in enumerate(documents):
            if not isinstance(doc, Document):
                raise TypeError(
                    f"Expected langchain_core.documents.Document at index {idx}, got {type(doc).__name__}"
                )

            cid = doc.metadata.get("chunk_id") if doc.metadata else None
            if not cid or not str(cid).strip():
                # Generate fallback chunk_id for defensive robustness
                fallback_cid = f"chunk_fallback_{uuid.uuid4().hex[:8]}"
                logger.warning(
                    f"Document at index {idx} missing valid 'chunk_id'. Generated fallback: {fallback_cid}"
                )
                cid = fallback_cid
                # Ensure metadata dict exists
                if doc.metadata is None:
                    doc.metadata = {}
                doc.metadata["chunk_id"] = fallback_cid

            cid_str = str(cid).strip()
            if cid_str not in seen_chunk_ids:
                seen_chunk_ids.add(cid_str)
                unique_docs.append(doc)

        if not unique_docs:
            return []

        # Step 2: Single document edge case (score directly, skip sorting)
        if len(unique_docs) == 1:
            single_doc = unique_docs[0]
            content = single_doc.page_content if single_doc.page_content else ""
            pair = [(clean_query, content)]
            scores = self._predict_with_guard(pair)
            score = scores[0] if scores else 0.0

            meta = dict(single_doc.metadata) if single_doc.metadata else {}
            original_rrf_rank = meta.get("original_rrf_rank", 1)
            original_rrf_score = meta.get("original_rrf_score", 0.0)

            meta["rerank_score"] = score
            meta["rerank_rank"] = 1
            meta["original_rrf_rank"] = original_rrf_rank
            meta["original_rrf_score"] = original_rrf_score

            annotated_doc = Document(page_content=single_doc.page_content, metadata=meta)
            return [(annotated_doc, score, 1)]

        # Step 3: Construct query-document pairs
        pairs: List[Tuple[str, str]] = [
            (clean_query, doc.page_content if doc.page_content else "")
            for doc in unique_docs
        ]

        # Step 4: Run batch prediction
        scores = self._predict_with_guard(pairs)

        # Step 5: Deterministic ranking with tie-breaking
        # Primary: score descending (-score)
        # Secondary: original RRF rank ascending
        # Tertiary: chunk_id ascending
        scored_items: List[Dict[str, Any]] = []
        for orig_idx, (doc, score) in enumerate(zip(unique_docs, scores)):
            meta = doc.metadata or {}
            orig_rrf_rank = meta.get("original_rrf_rank", orig_idx + 1)
            orig_rrf_score = meta.get("original_rrf_score", 0.0)
            cid = str(meta.get("chunk_id", f"doc_{orig_idx}"))

            scored_items.append({
                "doc": doc,
                "score": score,
                "orig_rrf_rank": orig_rrf_rank,
                "orig_rrf_score": orig_rrf_score,
                "chunk_id": cid,
            })

        sorted_items = sorted(
            scored_items,
            key=lambda x: (-x["score"], x["orig_rrf_rank"], x["chunk_id"]),
        )

        # Step 6: Format final top_k documents with annotated metadata
        final_results: List[Tuple[Document, float, int]] = []
        for final_rank, item in enumerate(sorted_items[:top_k], start=1):
            src_doc = item["doc"]
            meta = dict(src_doc.metadata) if src_doc.metadata else {}

            meta["rerank_score"] = item["score"]
            meta["rerank_rank"] = final_rank
            meta["original_rrf_rank"] = item["orig_rrf_rank"]
            meta["original_rrf_score"] = item["orig_rrf_score"]

            annotated_doc = Document(
                page_content=src_doc.page_content,
                metadata=meta,
            )
            final_results.append((annotated_doc, item["score"], final_rank))

        return final_results

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """
        Rerank candidates and return Top-K Document objects.

        Args:
            query: The search query string.
            documents: Candidate documents to rerank.
            top_k: Number of top documents to return (default: 5).

        Returns:
            List of reranked Document objects up to top_k.
        """
        results_with_scores = self.rerank_with_scores(query, documents, top_k=top_k)
        return [doc for doc, _, _ in results_with_scores]

    def __call__(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 5,
    ) -> List[Document]:
        """Callable convenience wrapper."""
        return self.rerank(query, documents, top_k=top_k)

