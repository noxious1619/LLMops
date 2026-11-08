"""
Integration test suite: Phase 5 Cross-Encoder Reranking into ConversationalRAG.

Validates the 13 required integration criteria:
1. Reranking enabled: Candidate pool reaches the reranker.
2. Reranking disabled: Existing Hybrid behaviour is preserved.
3. Final output contains <= 5 documents.
4. Original metadata preserved on final documents.
5. Cross-Encoder ordering is respected.
6. Singleton initialization: get_cached_reranker() is called once and reused.
7. Reranker failure with fallback enabled returns Hybrid Top-5.
8. Reranker failure with fallback disabled raises DocumentPortalException.
9. Conversational sessions and chat history continue working.
10. Query contextualization / rewriting continues working.
11. _format_docs receives final reranked documents.
12. FastAPI /chat endpoint schema remains backward compatible ({"answer": ...}).
13. Existing Dense and Hybrid modes remain unbroken.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from main import SESSIONS, app
from multi_doc_chat.exception.custom_exception import DocumentPortalException
from multi_doc_chat.models.models import ChatResponse
from multi_doc_chat.src.document_chat.hybrid_retriever import HybridRetriever
from multi_doc_chat.src.document_chat.reranker import CrossEncoderReranker
from multi_doc_chat.src.document_chat.retrieval import (
    ConversationalRAG,
    get_cached_reranker,
)


class MockLLM(Runnable[Any, str]):
    """Controllable mock LLM for deterministic RAG chain testing."""

    def __init__(self, fixed_answer: str = "Mocked LLM answer based on retrieved context."):
        self.invocations: List[Any] = []
        self.fixed_answer = fixed_answer

    def invoke(self, input_val: Any, config: Optional[Any] = None) -> str:
        self.invocations.append(input_val)
        return self.fixed_answer


class MockVectorStore:
    """Mock FAISS vectorstore with controllable docstore and retriever."""

    def __init__(self, docs: List[Document]):
        self.docs = docs
        self.index_to_docstore_id = {i: f"doc_{i}" for i in range(len(docs))}
        self.docstore = MagicMock()
        self.docstore.search.side_effect = lambda doc_id: self.docs[int(doc_id.split("_")[1])]

    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None):
        kwargs = search_kwargs or {}
        k = kwargs.get("k", len(self.docs))
        sliced_docs = self.docs[:k]

        class _MockDenseRetriever(Runnable[str, List[Document]]):
            def __init__(self, return_docs):
                self.return_docs = return_docs
                self.queries = []

            def invoke(self, query: str, config: Optional[Any] = None) -> List[Document]:
                self.queries.append(query)
                return list(self.return_docs)

        return _MockDenseRetriever(sliced_docs)


class MockReranker:
    """Controllable mock reranker for testing candidate passing and reordering."""

    def __init__(self, model_name: str = "mock-minilm", device: str = "cpu", batch_size: int = 5):
        self.model_name = model_name
        self.actual_device = device
        self.batch_size = batch_size
        self.rerank_calls: List[Dict[str, Any]] = []

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        self.rerank_calls.append({"query": query, "documents": list(documents), "top_k": top_k})
        # Reverse documents to demonstrate reordering
        reordered = list(reversed(documents))[:top_k]
        for idx, doc in enumerate(reordered, start=1):
            doc.metadata["rerank_score"] = float(10.0 - idx)
            doc.metadata["rerank_rank"] = idx
        return reordered


def make_doc(cid: str, text: str, source: str = "doc.txt", page: int = 1) -> Document:
    return Document(
        page_content=text,
        metadata={
            "chunk_id": cid,
            "source": source,
            "page": page,
            "content_hash": f"hash_{cid}",
        },
    )


@pytest.fixture
def mock_rag():
    """Create a ConversationalRAG instance with a mock LLM."""
    rag = ConversationalRAG(session_id="test_session")
    rag.llm = MockLLM()
    return rag


# ---------------------------------------------------------------------------
# Test Cases (1 to 13)
# ---------------------------------------------------------------------------

def test_01_reranking_enabled_candidate_pool_reaches_reranker(mock_rag, tmp_path):
    """1. Reranking enabled: Candidate pool reaches the reranker."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [make_doc(f"c{i}", f"Document text about topic {i}") for i in range(12)]
    mock_vs = MockVectorStore(mock_docs)
    mock_reranker = MockReranker()

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(
            index_path=str(faiss_dir),
            mode="hybrid",
            reranker_enabled=True,
            reranker_candidate_k=10,
        )
        mock_rag.reranker = mock_reranker
        mock_rag._build_lcel_chain()

    assert mock_rag.retrieval_mode == "hybrid + rerank"
    result = mock_rag.invoke("What is topic 5?")
    assert result == "Mocked LLM answer based on retrieved context."
    assert len(mock_reranker.rerank_calls) == 1
    call = mock_reranker.rerank_calls[0]
    assert call["top_k"] == 5
    assert len(call["documents"]) <= 10
    assert mock_rag.last_diagnostics["candidates_passed_to_reranker"] == len(call["documents"])


def test_02_reranking_disabled_hybrid_behavior_preserved(mock_rag, tmp_path):
    """2. Reranking disabled: Existing Hybrid behaviour is preserved."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [make_doc(f"c{i}", f"Document text about topic {i}") for i in range(8)]
    mock_vs = MockVectorStore(mock_docs)

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(
            index_path=str(faiss_dir),
            mode="hybrid",
            reranker_enabled=False,
        )

    assert mock_rag.retrieval_mode == "hybrid"
    assert mock_rag.reranker_enabled is False
    assert mock_rag.reranker is None

    res = mock_rag.invoke("Tell me about topic 2")
    assert res == "Mocked LLM answer based on retrieved context."
    assert mock_rag.last_diagnostics["retrieval_mode"] == "hybrid"
    assert mock_rag.last_diagnostics["candidates_passed_to_reranker"] == 0


def test_03_final_output_contains_no_more_than_five_docs(mock_rag, tmp_path):
    """3. Final output contains <= 5 documents."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [make_doc(f"c{i}", f"Document {i} details") for i in range(15)]
    mock_vs = MockVectorStore(mock_docs)
    mock_reranker = MockReranker()

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(
            index_path=str(faiss_dir),
            mode="hybrid",
            reranker_enabled=True,
            reranker_candidate_k=15,
        )
        mock_rag.reranker = mock_reranker
        mock_rag._build_lcel_chain()

    diag = mock_rag.invoke_with_diagnostics("Test query")["diagnostics"]
    assert len(diag["final_chunk_ids"]) <= 5
    assert len(diag["rerank_scores"]) <= 5


def test_04_original_metadata_preserved_on_final_documents(mock_rag, tmp_path):
    """4. Original metadata preserved on final documents."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [
        make_doc("chunk_alpha", "Content Alpha", source="alpha.pdf", page=4),
        make_doc("chunk_beta", "Content Beta", source="beta.pdf", page=7),
    ]
    mock_vs = MockVectorStore(mock_docs)
    mock_reranker = MockReranker()

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(
            index_path=str(faiss_dir),
            mode="hybrid",
            reranker_enabled=True,
        )
        mock_rag.reranker = mock_reranker
        mock_rag._build_lcel_chain()

    diag = mock_rag.invoke_with_diagnostics("Explain Alpha")["diagnostics"]
    assert "chunk_alpha" in diag["final_chunk_ids"] or "chunk_beta" in diag["final_chunk_ids"]
    assert "alpha.pdf" in diag["source_documents"] or "beta.pdf" in diag["source_documents"]


def test_05_cross_encoder_ordering_is_respected(mock_rag, tmp_path):
    """5. Cross-Encoder ordering is respected."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [
        make_doc("c1", "Low relevance doc"),
        make_doc("c2", "Medium relevance doc"),
        make_doc("c3", "High relevance doc"),
    ]
    mock_vs = MockVectorStore(mock_docs)

    class CustomOrderReranker(MockReranker):
        def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
            # Sort specifically so c3 is first, then c2, then c1
            order = {"c3": 3, "c2": 2, "c1": 1}
            sorted_docs = sorted(documents, key=lambda d: order.get(d.metadata["chunk_id"], 0), reverse=True)
            for idx, doc in enumerate(sorted_docs, start=1):
                doc.metadata["rerank_rank"] = idx
                doc.metadata["rerank_score"] = float(100 - idx)
            return sorted_docs[:top_k]

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid", reranker_enabled=True)
        mock_rag.reranker = CustomOrderReranker()
        mock_rag._build_lcel_chain()

    diag = mock_rag.invoke_with_diagnostics("Find most relevant")["diagnostics"]
    assert diag["final_chunk_ids"][0] == "c3"
    assert diag["final_chunk_ids"][1] == "c2"
    assert diag["final_chunk_ids"][2] == "c1"


def test_06_singleton_initialization_cached_across_instances():
    """6. Singleton initialization: get_cached_reranker() is called once and reused."""
    from multi_doc_chat.src.document_chat import retrieval
    # Reset singleton state for clean verification
    retrieval._CACHED_RERANKER = None

    with patch("multi_doc_chat.src.document_chat.retrieval.CrossEncoderReranker") as mock_init:
        dummy_instance = MagicMock()
        mock_init.return_value = dummy_instance

        r1 = get_cached_reranker()
        r2 = get_cached_reranker()
        r3 = get_cached_reranker()

        assert r1 is r2
        assert r2 is r3
        # Class constructor should have been invoked exactly once
        assert mock_init.call_count == 1


def test_07_reranker_failure_fallback_enabled_returns_hybrid_top5(mock_rag, tmp_path):
    """7. Reranker failure with fallback enabled returns Hybrid Top-5."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [make_doc(f"c{i}", f"Content {i}") for i in range(8)]
    mock_vs = MockVectorStore(mock_docs)

    class FailingReranker(MockReranker):
        def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
            raise RuntimeError("Simulated GPU OOM or model crash")

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(
            index_path=str(faiss_dir),
            mode="hybrid",
            reranker_enabled=True,
            fallback_to_hybrid=True,
        )
        mock_rag.reranker = FailingReranker()
        mock_rag._build_lcel_chain()

    # Should not raise exception; should fallback
    diag = mock_rag.invoke_with_diagnostics("Test with failing reranker")["diagnostics"]
    assert diag["fallback_activated"] is True
    assert diag["retrieval_mode"] == "hybrid (fallback)"
    assert len(diag["final_chunk_ids"]) <= 5


def test_08_reranker_failure_fallback_disabled_raises_exception(mock_rag, tmp_path):
    """8. Reranker failure with fallback disabled raises DocumentPortalException."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [make_doc(f"c{i}", f"Content {i}") for i in range(8)]
    mock_vs = MockVectorStore(mock_docs)

    class FailingReranker(MockReranker):
        def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
            raise RuntimeError("Fatal reranker crash")

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(
            index_path=str(faiss_dir),
            mode="hybrid",
            reranker_enabled=True,
            fallback_to_hybrid=False,
        )
        mock_rag.reranker = FailingReranker()
        mock_rag.reranker_fallback_to_hybrid = False
        mock_rag._build_lcel_chain()

    with pytest.raises(DocumentPortalException) as exc_info:
        mock_rag.invoke("Query that will trigger crash")

    assert "Invocation error in ConversationalRAG" in str(exc_info.value)


def test_09_conversational_sessions_and_chat_history_continue_working(mock_rag, tmp_path):
    """9. Conversational sessions and chat history continue working."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [make_doc("c1", "Redis cluster uses 16384 hash slots.")]
    mock_vs = MockVectorStore(mock_docs)
    mock_reranker = MockReranker()

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid", reranker_enabled=True)
        mock_rag.reranker = mock_reranker
        mock_rag._build_lcel_chain()

    history = [
        HumanMessage(content="What is Redis?"),
        AIMessage(content="Redis is an in-memory key-value data store."),
    ]
    ans = mock_rag.invoke("How does it partition data across clusters?", chat_history=history)
    assert ans == "Mocked LLM answer based on retrieved context."
    assert len(mock_rag.llm.invocations) >= 1


def test_10_query_contextualization_rewriting_continues_working(mock_rag, tmp_path):
    """10. Query contextualization / rewriting continues working."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [make_doc("c1", "Redis cluster slot resharding and failover.")]
    mock_vs = MockVectorStore(mock_docs)
    mock_reranker = MockReranker()

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid", reranker_enabled=True)
        mock_rag.reranker = mock_reranker
        mock_rag._build_lcel_chain()

    # Custom mock LLM that rewrites follow-up question
    mock_rag.llm = MockLLM(fixed_answer="How does Redis handle slot resharding during failover?")
    mock_rag._build_lcel_chain()

    history = [
        HumanMessage(content="Explain Redis clustering."),
        AIMessage(content="Redis clustering provides automatic sharding across multiple nodes."),
    ]
    diag = mock_rag.invoke_with_diagnostics("What happens during failover?", chat_history=history)["diagnostics"]
    assert "query" in diag
    assert len(diag["query"]) > 0


def test_11_format_docs_receives_final_reranked_documents(mock_rag, tmp_path):
    """11. _format_docs receives final reranked documents."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [
        make_doc("c1", "First doc content"),
        make_doc("c2", "Second doc content"),
    ]
    mock_vs = MockVectorStore(mock_docs)
    mock_reranker = MockReranker()

    formatted_docs_captured = []
    original_format = ConversationalRAG._format_docs

    def spy_format(docs):
        formatted_docs_captured.append(list(docs))
        return original_format(docs)

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"), \
         patch.object(ConversationalRAG, "_format_docs", side_effect=spy_format):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid", reranker_enabled=True)
        mock_rag.reranker = mock_reranker
        mock_rag._build_lcel_chain()
        mock_rag.invoke("Test doc format")

    assert len(formatted_docs_captured) == 1
    docs_sent = formatted_docs_captured[0]
    assert len(docs_sent) <= 5
    for d in docs_sent:
        assert "rerank_score" in d.metadata


def test_12_fastapi_chat_endpoint_schema_backward_compatible():
    """12. FastAPI /chat endpoint schema remains backward compatible ({"answer": ...})."""
    session_id = "test_fastapi_session"
    SESSIONS[session_id] = []

    session_faiss_dir = Path("faiss_index") / session_id
    session_faiss_dir.mkdir(parents=True, exist_ok=True)

    mock_docs = [make_doc("c1", "BGP border gateway protocol description.")]
    mock_vs = MockVectorStore(mock_docs)
    mock_reranker = MockReranker()

    try:
        with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
             patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"), \
             patch("multi_doc_chat.src.document_chat.retrieval.get_cached_reranker", return_value=mock_reranker), \
             patch("multi_doc_chat.utils.model_loader.ModelLoader.load_llm", return_value=MockLLM("FastAPI verified answer")):

            client = TestClient(app)
            response = client.post(
                "/chat",
                json={"session_id": session_id, "message": "What is BGP?"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert data["answer"] == "FastAPI verified answer"
        # Ensure no unexpected breaking keys in response model
        chat_resp = ChatResponse(**data)
        assert chat_resp.answer == "FastAPI verified answer"
    finally:
        if session_faiss_dir.exists():
            session_faiss_dir.rmdir()
        if session_id in SESSIONS:
            del SESSIONS[session_id]


def test_13_existing_dense_and_hybrid_modes_unbroken(mock_rag, tmp_path):
    """13. Existing Dense and Hybrid modes remain unbroken."""
    faiss_dir = tmp_path / "faiss_idx"
    faiss_dir.mkdir()

    mock_docs = [make_doc(f"c{i}", f"Data point {i}") for i in range(5)]
    mock_vs = MockVectorStore(mock_docs)

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):

        # 13a. Pure dense mode
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="dense")
        assert mock_rag.retrieval_mode == "dense"
        ans_dense = mock_rag.invoke("Dense query test")
        assert ans_dense == "Mocked LLM answer based on retrieved context."
        assert mock_rag.last_diagnostics["retrieval_mode"] == "dense"
        assert mock_rag.last_diagnostics["candidates_passed_to_reranker"] == 0

        # 13b. Pure hybrid mode (reranker disabled)
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid", reranker_enabled=False)
        assert mock_rag.retrieval_mode == "hybrid"
        ans_hybrid = mock_rag.invoke("Hybrid query test")
        assert ans_hybrid == "Mocked LLM answer based on retrieved context."
        assert mock_rag.last_diagnostics["retrieval_mode"] == "hybrid"
        assert mock_rag.last_diagnostics["candidates_passed_to_reranker"] == 0
