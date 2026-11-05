"""
Hybrid Retrieval Module combining Dense (FAISS) and Sparse (BM25) search via RRF.

Workflow:
1. Accept query.
2. Dense retrieval: FAISS retrieves Top-10 chunks.
3. Sparse retrieval: BM25 retrieves Top-10 chunks.
4. Reciprocal Rank Fusion: Combines both Top-10 lists using RRF (default k=60).
5. Final ranking: Deduplicates by chunk_id and returns Top-5 final Document chunks.
6. Metadata preservation: All original metadata, content, and chunk_ids remain intact.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from multi_doc_chat.src.document_chat.bm25_retriever import BM25Retriever
from multi_doc_chat.src.document_chat.rrf_fusion import (
    RRFResult,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


class HybridRetriever(Runnable[str, List[Document]]):
    """
    Hybrid Retriever executing two-stage hybrid retrieval:
    Dense semantic retrieval (FAISS) + Sparse lexical retrieval (BM25) fused via RRF.
    Inherits from LangChain Runnable to support native LCEL pipe operations (|).
    """

    def __init__(
        self,
        faiss_retriever: Any,
        bm25_retriever: Any,
        dense_k: int = 10,
        sparse_k: int = 10,
        final_k: int = 5,
        rrf_k: int = 60,
        fallback_to_dense: bool = False,
    ):
        """
        Initialize HybridRetriever with pre-configured dense and sparse retriever components.

        Args:
            faiss_retriever: Dense retriever (LangChain retriever, FAISS vectorstore, or callable).
            bm25_retriever: Sparse retriever (BM25Retriever, callable, or None).
            dense_k: Number of candidate chunks to fetch from FAISS (default: 10).
            sparse_k: Number of candidate chunks to fetch from BM25 (default: 10).
            final_k: Number of final chunks to return after fusion (default: 5).
            rrf_k: RRF smoothing constant (default: 60).
            fallback_to_dense: Whether to gracefully fall back to dense-only candidates
                               if BM25 is unavailable or fails (default: True).
        """
        if dense_k <= 0 or sparse_k <= 0 or final_k <= 0:
            raise ValueError("Retrieval sizes (dense_k, sparse_k, final_k) must be positive integers.")
        if rrf_k <= 0:
            raise ValueError("RRF parameter rrf_k must be a positive number.")

        self.faiss_retriever = faiss_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_k = dense_k
        self.sparse_k = sparse_k
        self.final_k = final_k
        self.rrf_k = rrf_k
        self.fallback_to_dense = fallback_to_dense

    def _fetch_dense(self, query: str) -> List[Document]:
        """Fetch top candidates from the dense FAISS retriever."""
        if self.faiss_retriever is None:
            return []

        docs: Any
        if hasattr(self.faiss_retriever, "invoke"):
            docs = self.faiss_retriever.invoke(query)
        elif hasattr(self.faiss_retriever, "get_relevant_documents"):
            docs = self.faiss_retriever.get_relevant_documents(query)
        elif hasattr(self.faiss_retriever, "similarity_search"):
            docs = self.faiss_retriever.similarity_search(query, k=self.dense_k)
        elif callable(self.faiss_retriever):
            docs = self.faiss_retriever(query)
        else:
            raise TypeError(
                f"Unsupported FAISS retriever type: {type(self.faiss_retriever).__name__}. "
                "Must support .invoke(), .get_relevant_documents(), .similarity_search(), or be callable."
            )

        if not isinstance(docs, list):
            raise TypeError(
                f"Dense FAISS retriever returned invalid type {type(docs).__name__}; expected list of Documents."
            )

        # Enforce maximum of dense_k candidates
        return docs[:self.dense_k]

    def _fetch_sparse(self, query: str) -> List[Document]:
        """Fetch top candidates from the sparse BM25 retriever."""
        if self.bm25_retriever is None:
            return []

        try:
            docs: Any
            if hasattr(self.bm25_retriever, "retrieve_documents"):
                docs = self.bm25_retriever.retrieve_documents(query, top_k=self.sparse_k)
            elif hasattr(self.bm25_retriever, "retrieve"):
                raw_entries = self.bm25_retriever.retrieve(query, top_k=self.sparse_k)
                if not isinstance(raw_entries, list):
                    raise TypeError(
                        f"Sparse BM25 retriever returned invalid type {type(raw_entries).__name__}; expected list."
                    )
                docs = [
                    Document(
                        page_content=entry["original_text"],
                        metadata=dict(entry["metadata"]),
                    )
                    for entry in raw_entries
                ]
            elif hasattr(self.bm25_retriever, "invoke"):
                docs = self.bm25_retriever.invoke(query)
            elif callable(self.bm25_retriever):
                docs = self.bm25_retriever(query)
            else:
                raise TypeError(
                    f"Unsupported BM25 retriever type: {type(self.bm25_retriever).__name__}. "
                    "Must support .retrieve_documents(), .retrieve(), .invoke(), or be callable."
                )

            if not isinstance(docs, list):
                raise TypeError(
                    f"Sparse BM25 retriever returned invalid type {type(docs).__name__}; expected list of Documents."
                )

            # Enforce maximum of sparse_k candidates
            return docs[:self.sparse_k]

        except Exception as e:
            if self.fallback_to_dense:
                logger.warning(
                    f"BM25 retrieval failed ({type(e).__name__}: {str(e)}); "
                    "falling back to dense-only candidates."
                )
                return []
            raise

    def retrieve_with_scores(self, query: str) -> List[RRFResult]:
        """
        Execute hybrid retrieval and return fused candidates with RRF scores.

        Args:
            query: User search query string.

        Returns:
            List of RRFResult namedtuples (document, score, rank) up to final_k.
        """
        if not query or not query.strip():
            return []

        clean_query = query.strip()
        dense_docs = self._fetch_dense(clean_query)
        sparse_docs = self._fetch_sparse(clean_query)

        # Pass both ranked lists to RRF fusion
        fused = reciprocal_rank_fusion([dense_docs, sparse_docs], k=self.rrf_k)

        # Return Top-final_k chunks
        return fused[:self.final_k]

    def retrieve(self, query: str) -> List[Document]:
        """
        Execute full hybrid retrieval pipeline and return Top-5 Document chunks.

        Args:
            query: User search query string.

        Returns:
            List of Top-5 Document objects with preserved metadata.
        """
        results_with_scores = self.retrieve_with_scores(query)
        return [r.document for r in results_with_scores]

    def invoke(self, input: str, config: Optional[Any] = None) -> List[Document]:
        """LangChain LCEL runnable interface compatibility."""
        return self.retrieve(input)

    @classmethod
    def from_saved(
        cls,
        faiss_index_dir: Path | str = "faiss_index/benchmark_70docs_v1",
        bm25_index_dir: Optional[Path | str] = "data/bm25_index",
        embeddings: Optional[Any] = None,
        dense_k: int = 10,
        sparse_k: int = 10,
        final_k: int = 5,
        rrf_k: int = 60,
        search_type: str = "similarity",
        fallback_to_dense: bool = True,
    ) -> HybridRetriever:
        """
        Factory to load existing persistent FAISS and BM25 indexes in read-only mode.
        If bm25_index_dir does not exist or index is session-specific, builds an
        in-memory session-matched BM25Retriever directly from FAISS docstore chunks.

        Args:
            faiss_index_dir: Path to directory containing existing index.faiss and index.pkl.
            bm25_index_dir: Optional path to directory containing bm25_index.pkl and bm25_manifest.json.
            embeddings: Optional embeddings instance (e.g. ModelLoader().load_embeddings()).
            dense_k: Top-K dense results to retrieve (default: 10).
            sparse_k: Top-K sparse results to retrieve (default: 10).
            final_k: Top-K fused results to return (default: 5).
            rrf_k: RRF smoothing constant (default: 60).
            search_type: FAISS search type (default: "similarity").
            fallback_to_dense: Whether to fall back to dense-only if BM25 fails.

        Returns:
            Configured, ready-to-query HybridRetriever instance.
        """
        faiss_path = Path(faiss_index_dir)
        if not faiss_path.exists():
            raise FileNotFoundError(f"FAISS index directory not found at {faiss_path}")

        # Lazy load embeddings if not provided
        if embeddings is None:
            from multi_doc_chat.utils.model_loader import ModelLoader
            embeddings = ModelLoader().load_embeddings()

        from langchain_community.vectorstores import FAISS

        vectorstore = FAISS.load_local(
            str(faiss_path),
            embeddings,
            index_name="index",
            allow_dangerous_deserialization=True,
        )
        faiss_retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": dense_k},
        )

        bm25_retriever: Optional[BM25Retriever] = None
        bm25_path = Path(bm25_index_dir) if bm25_index_dir else None

        # Check if pre-built BM25 index bundle exists on disk
        if bm25_path and (bm25_path / "bm25_index.pkl").exists() and (bm25_path / "bm25_manifest.json").exists():
            try:
                bm25_retriever = BM25Retriever(index_dir=bm25_path)
            except Exception as e:
                if not fallback_to_dense:
                    raise
                logger.warning(f"Failed to load persisted BM25 index from {bm25_path}: {e}")

        # Otherwise, dynamically construct session-matched BM25 index from FAISS docstore
        if bm25_retriever is None:
            try:
                session_docs: List[Document] = []
                if hasattr(vectorstore, "docstore") and hasattr(vectorstore, "index_to_docstore_id"):
                    for doc_id in vectorstore.index_to_docstore_id.values():
                        doc = vectorstore.docstore.search(doc_id)
                        if isinstance(doc, Document):
                            session_docs.append(doc)

                if session_docs:
                    bm25_retriever = BM25Retriever.from_documents(session_docs)
                else:
                    logger.warning("No documents found in FAISS docstore for BM25 construction.")
            except Exception as e:
                if not fallback_to_dense:
                    raise
                logger.warning(f"Failed to construct session BM25 retriever: {e}")

        return cls(
            faiss_retriever=faiss_retriever,
            bm25_retriever=bm25_retriever,
            dense_k=dense_k,
            sparse_k=sparse_k,
            final_k=final_k,
            rrf_k=rrf_k,
            fallback_to_dense=fallback_to_dense,
        )
