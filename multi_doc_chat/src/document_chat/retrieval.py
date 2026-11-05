from __future__ import annotations

import os
import sys
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
from multi_doc_chat.utils.config_loader import load_config
from multi_doc_chat.utils.model_loader import ModelLoader


class ConversationalRAG:
    """
    LCEL-based Conversational RAG with configurable Hybrid (FAISS + BM25 via RRF)
    and Dense (FAISS-only) retrieval modes.

    Usage:
        rag = ConversationalRAG(session_id="abc")
        # Hybrid retrieval (default):
        rag.load_retriever(index_path="faiss_index/abc", mode="hybrid")
        # Or dense-only retrieval:
        rag.load_retriever(index_path="faiss_index/abc", mode="dense")

        answer = rag.invoke("What is ...?", chat_history=[])
    """

    def __init__(self, session_id: Optional[str], retriever=None):
        try:
            self.session_id = session_id
            self.retrieval_mode = "uninitialized"

            # Load config
            try:
                self.config = load_config()
            except Exception:
                self.config = {}

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

            log.info("ConversationalRAG initialized", session_id=self.session_id)
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
    ):
        """
        Load vectorstore and configure retrieval mode (Hybrid or Dense).

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
            env_mode = os.getenv("RETRIEVAL_MODE")
            chosen_mode = (mode or env_mode or retriever_cfg.get("mode", "hybrid")).strip().lower()

            if chosen_mode not in {"hybrid", "dense"}:
                raise ValueError(
                    f"Invalid retrieval mode '{chosen_mode}'. Supported modes are 'hybrid' and 'dense'."
                )

            final_k = k or retriever_cfg.get("top_k", 5)
            d_k = dense_k or retriever_cfg.get("dense_k", 10)
            s_k = sparse_k or retriever_cfg.get("sparse_k", 10)
            r_k = rrf_k or retriever_cfg.get("rrf_k", 60)
            allow_fallback = (
                fallback_to_dense
                if fallback_to_dense is not None
                else retriever_cfg.get("fallback_to_dense", True)
            )
            stype = search_type or retriever_cfg.get("search_type", "similarity")

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
                    self.retrieval_mode = "hybrid"
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
        )

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """Invoke the LCEL pipeline."""
        try:
            if self.chain is None:
                raise DocumentPortalException(
                    "RAG chain not initialized. Call load_retriever() before invoke().", sys
                )
            chat_history = chat_history or []
            payload = {"input": user_input, "chat_history": chat_history}
            answer = self.chain.invoke(payload)
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
            log.info(
                "Chain invoked successfully",
                session_id=self.session_id,
                user_input=user_input,
                answer_preview=str(answer)[:150],
            )
            return answer
        except Exception as e:
            log.error("Failed to invoke ConversationalRAG", error=str(e))
            raise DocumentPortalException("Invocation error in ConversationalRAG", sys)

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

            # 2) Retrieve docs for rewritten question (supports Runnable, BaseRetriever, or callable)
            retriever_runnable = (
                self.retriever
                if isinstance(self.retriever, Runnable)
                else RunnableLambda(self.retriever.invoke)
            )
            retrieve_docs = question_rewriter | retriever_runnable | self._format_docs

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
