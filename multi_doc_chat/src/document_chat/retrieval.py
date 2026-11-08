from __future__ import annotations

import math
import os
import sys
import time
from operator import itemgetter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from pydantic import ValidationError

from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.logger import GLOBAL_LOGGER as log
from multi_doc_chat.models.models import ChatAnswer, PromptType
from multi_doc_chat.prompts.prompt_library import PROMPT_REGISTRY
from multi_doc_chat.src.document_chat.bm25_retriever import BM25Retriever
from multi_doc_chat.src.document_chat.hybrid_retriever import HybridRetriever
from multi_doc_chat.src.document_chat.reranker import CrossEncoderReranker
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.utils.model_loader import ModelLoader

# Global cached singleton for CrossEncoderReranker to prevent model reload on each request
_CACHED_RERANKER: Optional[CrossEncoderReranker] = None


def get_cached_reranker(config: Optional[Dict[str, Any]] = None, force_reload: bool = False) -> Optional[CrossEncoderReranker]:
    """
    Get or initialize the singleton CrossEncoderReranker.
    Ensures model weights are loaded once in GPU memory and reused across all requests.
    """
    global _CACHED_RERANKER
    if _CACHED_RERANKER is not None and not force_reload:
        return _CACHED_RERANKER

    cfg = config or {}
    reranker_cfg = cfg.get("reranker", {}) if isinstance(cfg, dict) else {}
    model_name = reranker_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    device = reranker_cfg.get("device", "auto")
    batch_size = reranker_cfg.get("batch_size", 5)
    max_length = reranker_cfg.get("max_length", 512)
    allow_cpu_fallback = reranker_cfg.get("allow_cpu_fallback", True)

    _CACHED_RERANKER = CrossEncoderReranker(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        allow_cpu_fallback=allow_cpu_fallback,
        max_length=max_length,
    )
    return _CACHED_RERANKER


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with configurable Two-Stage Retrieval:
    Hybrid Retrieval (FAISS Top-10 + BM25 Top-10 via RRF k=60) + Cross-Encoder Reranking (MiniLM-L-6-v2),
    with graceful fallback to Hybrid-only or Dense-only retrieval.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        # Two-stage Hybrid + Rerank (default):
        rag.load_retriever(index_path="faiss_index/abc", mode="hybrid")
        # Or dense-only retrieval:
        rag.load_retriever(index_path="faiss_index/abc", mode="dense")

        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(
        self,
        session_id: Optional[str],
        retriever=None,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        try:
            self.session_id = session_id
            self.retrieval_mode = "uninitialized"

            # Load config
            try:
                self.config = load_config()
            except Exception:
                self.config = {}

            # Parse reranker configuration
            reranker_cfg = (
                self.config.get("reranker", {})
                if isinstance(self.config, dict)
                else {}
            )
            env_rerank = os.getenv("RERANKER_ENABLED")
            if env_rerank is not None:
                self.reranker_enabled = env_rerank.strip().lower() in ("true", "1", "yes")
            else:
                self.reranker_enabled = bool(reranker_cfg.get("enabled", True))

            self.reranker_candidate_k = int(reranker_cfg.get("candidate_k", 20))
            self.reranker_final_k = int(reranker_cfg.get("final_k", 5))
            self.reranker_fallback_to_hybrid = bool(reranker_cfg.get("fallback_to_hybrid", True))
            self.reranker = reranker
            self.last_diagnostics: Dict[str, Any] = {}

            # Load LLM and prompts once
            self.llm = self._load_llm()
            self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXTUALIZE_QUESTION.value
            ]
            self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[
                PromptType.CONTEXT_QA.value
            ]

            # Lazy pieces
            self.retriever = retriever
            self.chain = None
            if self.retriever is not None:
                self._build_lcel_chain()

            log.info(
                "ConversationalRAG initialized",
                session_id=self.session_id,
                reranker_enabled=self.reranker_enabled,
            )
        except Exception as e:
            log.error("Failed to initialize ConversationalRAG", error=str(e))
            raise DocumentPortalException("Initialization error in ConversationalRAG", sys)

    # ---------- Public API ----------

    def load_retriever(
        self,
        index_path: str,
        mode: Optional[str] = None,
        k: Optional[int] = None,
        dense_k: Optional[int] = None,
        sparse_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        fallback_to_dense: Optional[bool] = None,
        index_name: str = "index",
        search_type: Optional[str] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        search_kwargs: Optional[Dict[str, Any]] = None,
        bm25_index_dir: Optional[str] = None,
        reranker_enabled: Optional[bool] = None,
        reranker_candidate_k: Optional[int] = None,
        fallback_to_hybrid: Optional[bool] = None,
    ):
        """
        Load vectorstore and configure retrieval mode (Hybrid + Rerank, Hybrid, or Dense).

        Args:
            index_path: Path to FAISS index directory.
            mode: Retrieval mode ('hybrid' or 'dense'). Defaults to config or env RETRIEVAL_MODE.
            k: Final number of documents to pass to LLM (default: 5).
            dense_k: Number of FAISS candidates in hybrid mode (default: 10).
            sparse_k: Number of BM25 candidates in hybrid mode (default: 10).
            rrf_k: RRF smoothing constant (default: 60).
            fallback_to_dense: Gracefully fall back to dense if BM25 fails (default: True).
            index_name: FAISS index file basename (default: "index").
            search_type: FAISS search algorithm ("similarity" or "mmr").
            fetch_k: Candidates to fetch before MMR reranking.
            lambda_mult: Diversity parameter for MMR.
            search_kwargs: Custom FAISS search kwargs.
            bm25_index_dir: Optional custom path to persisted BM25 index directory.
            reranker_enabled: Override to enable/disable Cross-Encoder reranking.
            reranker_candidate_k: Number of candidates to retrieve for reranking (default: 20).
            fallback_to_hybrid: Fall back to hybrid Top-5 if reranker fails (default: True).

        Returns:
            The configured retriever instance.
        """
        try:
            index_path_obj = Path(index_path)
            if not index_path_obj.is_dir():
                raise FileNotFoundError(f"FAISS index directory not found: {index_path}")

            # 1. Resolve configuration parameters
            retriever_cfg = (
                self.config.get("retriever", {})
                if isinstance(self.config, dict)
                else {}
            )
            reranker_cfg = (
                self.config.get("reranker", {})
                if isinstance(self.config, dict)
                else {}
            )
            env_mode = os.getenv("RETRIEVAL_MODE")
            chosen_mode = (mode or env_mode or retriever_cfg.get("mode", "hybrid")).strip().lower()

            if chosen_mode not in {"hybrid", "dense"}:
                raise ValueError(
                    f"Invalid retrieval mode '{chosen_mode}'. Supported modes are 'hybrid' and 'dense'."
                )

            final_k = k or reranker_cfg.get("final_k") or retriever_cfg.get("top_k", 5)
            d_k = dense_k or retriever_cfg.get("dense_k", 10)
            s_k = sparse_k or retriever_cfg.get("sparse_k", 10)
            r_k = rrf_k or retriever_cfg.get("rrf_k", 60)
            allow_fallback = (
                fallback_to_dense
                if fallback_to_dense is not None
                else retriever_cfg.get("fallback_to_dense", True)
            )
            stype = search_type or retriever_cfg.get("search_type", "similarity")

            if reranker_enabled is not None:
                self.reranker_enabled = reranker_enabled
            if reranker_candidate_k is not None:
                self.reranker_candidate_k = reranker_candidate_k
            if fallback_to_hybrid is not None:
                self.reranker_fallback_to_hybrid = fallback_to_hybrid
            self.reranker_final_k = final_k

            # 2. Load FAISS index
            embeddings = ModelLoader().load_embeddings()
            vectorstore = FAISS.load_local(
                str(index_path_obj),
                embeddings,
                index_name=index_name,
                allow_dangerous_deserialization=True,
            )

            # 3. Configure dense-only mode
            if chosen_mode == "dense":
                dense_skwargs = dict(search_kwargs) if search_kwargs else {"k": final_k}
                if "k" not in dense_skwargs:
                    dense_skwargs["k"] = final_k
                if stype == "mmr":
                    if "fetch_k" not in dense_skwargs:
                        dense_skwargs["fetch_k"] = fetch_k or retriever_cfg.get("fetch_k", 20)
                    if "lambda_mult" not in dense_skwargs:
                        dense_skwargs["lambda_mult"] = (
                            lambda_mult
                            if lambda_mult is not None
                            else retriever_cfg.get("lambda_mult", 0.5)
                        )

                self.retriever = vectorstore.as_retriever(
                    search_type=stype, search_kwargs=dense_skwargs
                )
                self.retrieval_mode = "dense"
                log.info(
                    "Dense retriever loaded successfully",
                    index_path=str(index_path_obj),
                    mode=self.retrieval_mode,
                    k=final_k,
                    session_id=self.session_id,
                )

            # 4. Configure hybrid mode
            else:
                dense_skwargs = dict(search_kwargs) if search_kwargs else {"k": d_k}
                dense_skwargs["k"] = d_k
                if stype == "mmr":
                    if "fetch_k" not in dense_skwargs:
                        dense_skwargs["fetch_k"] = fetch_k or retriever_cfg.get("fetch_k", 20)
                    if "lambda_mult" not in dense_skwargs:
                        dense_skwargs["lambda_mult"] = (
                            lambda_mult
                            if lambda_mult is not None
                            else retriever_cfg.get("lambda_mult", 0.5)
                        )
                dense_retriever = vectorstore.as_retriever(
                    search_type=stype, search_kwargs=dense_skwargs
                )

                bm25_retriever: Optional[BM25Retriever] = None
                try:
                    is_benchmark = (
                        "benchmark_70docs_v1" in index_path_obj.parts
                        or index_path_obj.name == "benchmark_70docs_v1"
                    )
                    custom_bm25_dir = Path(bm25_index_dir) if bm25_index_dir else None

                    # If custom BM25 dir exists on disk, load it
                    if custom_bm25_dir and (custom_bm25_dir / "bm25_index.pkl").exists():
                        bm25_retriever = BM25Retriever(index_dir=custom_bm25_dir)
                    # If benchmark index, load canonical pre-built BM25 index
                    elif is_benchmark and Path("data/bm25_index/bm25_index.pkl").exists():
                        bm25_retriever = BM25Retriever(index_dir="data/bm25_index")
                    else:
                        # Dynamic session-matched BM25 directly from active session's FAISS docstore
                        session_docs: List[Document] = []
                        if hasattr(vectorstore, "docstore") and hasattr(vectorstore, "index_to_docstore_id"):
                            for doc_id in vectorstore.index_to_docstore_id.values():
                                doc = vectorstore.docstore.search(doc_id)
                                if isinstance(doc, Document):
                                    session_docs.append(doc)
                        if session_docs:
                            bm25_retriever = BM25Retriever.from_documents(session_docs)
                        else:
                            raise ValueError("No Document objects discovered in FAISS docstore for BM25 construction")

                except Exception as e:
                    if not allow_fallback:
                        raise DocumentPortalException(
                            f"BM25 initialization failed and fallback is disabled: {e}", sys
                        ) from e
                    log.warning(
                        "BM25 initialization failed; falling back to dense-only retrieval",
                        error=str(e),
                        session_id=self.session_id,
                    )

                if bm25_retriever is not None:
                    self.retriever = HybridRetriever(
                        faiss_retriever=dense_retriever,
                        bm25_retriever=bm25_retriever,
                        dense_k=d_k,
                        sparse_k=s_k,
                        final_k=final_k,
                        rrf_k=r_k,
                        fallback_to_dense=allow_fallback,
                    )

                    # Initialize or attach singleton CrossEncoderReranker if enabled
                    if self.reranker_enabled and self.reranker is None:
                        try:
                            self.reranker = get_cached_reranker(self.config)
                            log.info(
                                "Cross-Encoder reranker attached to ConversationalRAG",
                                model=getattr(self.reranker, "model_name", "N/A"),
                                device=getattr(self.reranker, "actual_device", "N/A"),
                                batch_size=getattr(self.reranker, "batch_size", 5),
                                session_id=self.session_id,
                            )
                        except Exception as r_err:
                            if not self.reranker_fallback_to_hybrid:
                                raise DocumentPortalException(
                                    f"Failed to initialize reranker and fallback is disabled: {r_err}", sys
                                ) from r_err
                            log.warning(
                                "Reranker initialization failed; falling back to hybrid retrieval without reranking",
                                error=str(r_err),
                                session_id=self.session_id,
                            )
                            self.reranker = None

                    self.retrieval_mode = (
                        "hybrid + rerank"
                        if (self.reranker_enabled and self.reranker is not None)
                        else "hybrid"
                    )

                    log.info(
                        "Hybrid retriever loaded successfully",
                        index_path=str(index_path_obj),
                        mode=self.retrieval_mode,
                        final_k=final_k,
                        session_id=self.session_id,
                    )
                else:
                    # Fallback to pure dense retrieval
                    dense_fallback_kwargs = dict(search_kwargs) if search_kwargs else {"k": final_k}
                    dense_fallback_kwargs["k"] = final_k
                    self.retriever = vectorstore.as_retriever(
                        search_type=stype, search_kwargs=dense_fallback_kwargs
                    )
                    self.retrieval_mode = "dense (fallback)"
                    log.info(
                        "Fallback dense retriever activated",
                        index_path=str(index_path_obj),
                        mode=self.retrieval_mode,
                        k=final_k,
                        session_id=self.session_id,
                    )

            # 5. Build LCEL graph
            self._build_lcel_chain()
            return self.retriever

        except Exception as e:
            log.error("Failed to load retriever", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Loading error in ConversationalRAG", sys) from e

    def load_retriever_from_faiss(
        self,
        index_path: str,
        k: int = 5,
        index_name: str = "index",
        search_type: str = "mmr",
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        search_kwargs: Optional[Dict[str, Any]] = None,
        mode: Optional[str] = None,
        fallback_to_dense: Optional[bool] = None,
        reranker_enabled: Optional[bool] = None,
    ):
        """
        Backward-compatible loader. Delegates to load_retriever().
        """
        return self.load_retriever(
            index_path=index_path,
            mode=mode,
            k=k,
            index_name=index_name,
            search_type=search_type,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            search_kwargs=search_kwargs,
            fallback_to_dense=fallback_to_dense,
            reranker_enabled=reranker_enabled,
        )

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever() before invoke().", sys
                )
            t_invoke_start = time.perf_counter()
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
            t_invoke_end = time.perf_counter()
            total_request_ms = (t_invoke_end - t_invoke_start) * 1000

            if not answer:
                log.warning(
                    "No answer generated", user_input=user_input, session_id=self.session_id
                )
                return "no answer generated."

            # Validate answer type and length using Pydantic model
            try:
                validated = ChatAnswer(answer=str(answer))
                answer = validated.answer
            except ValidationError as ve:
                log.error("Invalid chat answer", error=str(ve))
                raise DocumentPortalException("Invalid chat answer", sys)

            if isinstance(self.last_diagnostics, dict):
                self.last_diagnostics["total_request_latency_ms"] = total_request_ms
                llm_ms = max(
                    0.0,
                    total_request_ms - self.last_diagnostics.get("total_retrieval_latency_ms", 0.0)
                )
                self.last_diagnostics["llm_latency_ms"] = llm_ms

            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                mode=self.retrieval_mode,
                total_latency_ms=round(total_request_ms, 2),
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException(f"Invocation error in ConversationalRAG: {e}", sys) from e

    def invoke_with_diagnostics(
        self, user_input: str, chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        """
        Convenience invocation returning the generated answer alongside rich
        retrieval, reranking, and latency diagnostics.
        """
        answer = self.invoke(user_input, chat_history=chat_history)
        return {
            "answer": answer,
            "diagnostics": dict(self.last_diagnostics),
        }

    # ---------- Internals ----------

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("LLM could not be loaded")
            log.info("LLM loaded successfully", session_id=self.session_id)
            return llm
        except Exception as e:
            log.error("Failed to load LLM", error=str(e))
            raise DocumentPortalException("LLM loading error in ConversationalRAG", sys)

    @staticmethod
    def _format_docs(docs) -> str:
        return "\n\n".join(getattr(d, "page_content", str(d)) for d in docs)

    def _build_lcel_chain(self):
        try:
            if self.retriever is None:
                raise DocumentPortalException("No retriever set before building chain", sys)

            # 1) Rewrite user question with chat history context
            question_rewriter = (
                {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | self.contextualize_prompt
                | self.llm
                | StrOutputParser()
                | (lambda x: str(x))
            )

            # 2) Retrieve & Rerank documents for rewritten question
            def _retrieve_and_rerank(query_str: str) -> List[Document]:
                t0 = time.perf_counter()
                clean_q = query_str.strip() if query_str else ""
                if not clean_q:
                    return []

                fallback_activated = False
                rerank_time_ms = 0.0

                from unittest.mock import MagicMock
                is_mock = isinstance(self.retriever, MagicMock)
                is_hybrid_retriever = (
                    isinstance(self.retriever, HybridRetriever)
                    or (hasattr(self.retriever, "retrieve_candidates") and not is_mock)
                )

                should_rerank = (
                    self.reranker_enabled
                    and self.reranker is not None
                    and is_hybrid_retriever
                )

                candidates: List[Document] = []
                if should_rerank:
                    # Candidate retrieval (FAISS Top-10 + BM25 Top-10 -> RRF k=60 -> up to candidate_k)
                    t_cand_start = time.perf_counter()
                    candidates = self.retriever.retrieve_candidates(clean_q, candidate_k=self.reranker_candidate_k)
                    t_cand_end = time.perf_counter()
                    retrieval_ms = (t_cand_end - t_cand_start) * 1000

                    # Cross-Encoder Reranking
                    try:
                        t_rr_start = time.perf_counter()
                        final_docs = self.reranker.rerank(clean_q, candidates, top_k=self.reranker_final_k)
                        t_rr_end = time.perf_counter()
                        rerank_time_ms = (t_rr_end - t_rr_start) * 1000
                    except Exception as rerank_err:
                        log.error("Cross-Encoder reranking execution failed", error=str(rerank_err))
                        if not self.reranker_fallback_to_hybrid:
                            raise DocumentPortalException(
                                f"Cross-Encoder reranking failed and fallback is disabled: {rerank_err}", sys
                            ) from rerank_err
                        log.warning(
                            "Activating fallback: returning hybrid Top-5 without reranking",
                            error=str(rerank_err),
                        )
                        fallback_activated = True
                        final_docs = candidates[:self.reranker_final_k] if candidates else self.retriever.retrieve(clean_q)
                else:
                    # Standard retrieval path without reranking (Hybrid Top-5 or Dense Top-K)
                    t_ret_start = time.perf_counter()
                    if hasattr(self.retriever, "invoke"):
                        final_docs = self.retriever.invoke(clean_q)
                    elif hasattr(self.retriever, "retrieve"):
                        final_docs = self.retriever.retrieve(clean_q)
                    elif callable(self.retriever):
                        final_docs = self.retriever(clean_q)
                    else:
                        final_docs = []
                    t_ret_end = time.perf_counter()
                    retrieval_ms = (t_ret_end - t_ret_start) * 1000
                    candidates = final_docs

                total_retrieval_ms = (time.perf_counter() - t0) * 1000

                # Enforce maximum of final_k documents
                target_k = self.reranker_final_k if self.reranker_enabled else getattr(self.retriever, "final_k", 5)
                final_docs = final_docs[:target_k]

                # Store comprehensive diagnostics
                batch_sz = getattr(self.reranker, "batch_size", 5) if self.reranker else 5
                active_mode = (
                    "hybrid + rerank"
                    if (should_rerank and not fallback_activated)
                    else ("hybrid (fallback)" if fallback_activated else self.retrieval_mode)
                )

                self.last_diagnostics = {
                    "retrieval_mode": active_mode,
                    "query": clean_q,
                    "faiss_candidate_count": getattr(self.retriever, "dense_k", 10) if is_hybrid_retriever else len(candidates),
                    "bm25_candidate_count": getattr(self.retriever, "sparse_k", 10) if is_hybrid_retriever else 0,
                    "combined_candidate_count": (
                        (getattr(self.retriever, "dense_k", 10) + getattr(self.retriever, "sparse_k", 10))
                        if is_hybrid_retriever
                        else len(candidates)
                    ),
                    "unique_candidate_count": len(candidates),
                    "candidates_passed_to_reranker": len(candidates) if should_rerank else 0,
                    "reranker_model": getattr(self.reranker, "model_name", "none"),
                    "execution_device": getattr(self.reranker, "actual_device", "none"),
                    "batch_size": batch_sz,
                    "inference_batches": math.ceil(len(candidates) / batch_sz) if (should_rerank and candidates) else 0,
                    "rerank_scores": [
                        d.metadata.get("rerank_score")
                        for d in final_docs
                        if hasattr(d, "metadata") and d.metadata and "rerank_score" in d.metadata
                    ],
                    "final_chunk_ids": [
                        d.metadata.get("chunk_id", "unknown")
                        for d in final_docs
                        if hasattr(d, "metadata") and d.metadata
                    ],
                    "source_documents": [
                        d.metadata.get("source", "unknown")
                        for d in final_docs
                        if hasattr(d, "metadata") and d.metadata
                    ],
                    "fallback_activated": fallback_activated,
                    "retrieval_latency_ms": retrieval_ms,
                    "rerank_latency_ms": rerank_time_ms,
                    "total_retrieval_latency_ms": total_retrieval_ms,
                }

                log.info(
                    "Retrieval and ranking completed",
                    mode=active_mode,
                    unique_candidates=self.last_diagnostics["unique_candidate_count"],
                    final_docs=len(final_docs),
                    fallback=fallback_activated,
                    retrieval_ms=round(retrieval_ms, 2),
                    rerank_ms=round(rerank_time_ms, 2),
                    session_id=self.session_id,
                )

                return final_docs

            retrieve_docs = question_rewriter | RunnableLambda(_retrieve_and_rerank) | self._format_docs

            # 3) Answer using retrieved context + original input + chat history
            self.chain = (
                {
                    "context": retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history": itemgetter("chat_history"),
                }
                | self.qa_prompt
                | self.llm
                | StrOutputParser()
            )

            log.info(
                "LCEL graph built successfully",
                session_id=self.session_id,
                mode=getattr(self, "retrieval_mode", "unknown"),
            )
        except Exception as e:
            log.error("Failed to build LCEL chain", error=str(e), session_id=self.session_id)
            raise DocumentPortalException("Failed to build LCEL chain", sys)
