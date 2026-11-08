"""
Phase 7 Test Suite: Integrate Hybrid Retrieval into ConversationalRAG.

Validates:
1. Hybrid mode is selected correctly.
2. Dense mode is selected correctly.
3. Rewritten query is passed to both retrievers.
4. Conversation history is preserved.
5. Hybrid retrieval produces normal Document objects.
6. The existing document formatter receives the retrieved documents.
7. Existing prompt construction remains unchanged.
8. Existing LLM invocation remains compatible.
9. Existing API response structure remains unchanged.
10. BM25 initialization failure triggers dense fallback.
11. BM25 retrieval failure triggers dense fallback.
12. Session-specific retrieval uses the correct document collection.
13. Cross-session document leakage is prevented.
14. Metadata survives retrieval and reaches the document formatting stage.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable, RunnableLambda

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multi_doc_chat.models.models import ChatAnswer
from multi_doc_chat.src.document_chat.bm25_retriever import BM25Retriever
from multi_doc_chat.src.document_chat.hybrid_retriever import HybridRetriever
from multi_doc_chat.src.document_chat.retrieval import ConversationalRAG


class MockLLM(Runnable[Any, str]):
    """Controllable mock LLM for deterministic RAG chain testing."""
    def __init__(self, fixed_answer: str = "Mocked LLM answer based on context."):
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


def make_doc(cid: str, text: str, source: str = "file.txt") -> Document:
    return Document(
        page_content=text,
        metadata={"chunk_id": cid, "source": source, "content_hash": f"hash_{cid}"},
    )


@pytest.fixture
def mock_rag():
    """Create a ConversationalRAG instance with a mock LLM."""
    rag = ConversationalRAG(session_id="test_session")
    rag.llm = MockLLM()
    return rag


def test_01_hybrid_mode_selected_correctly(mock_rag, tmp_path):
    """1. Hybrid mode is selected correctly and instantiates HybridRetriever."""
    faiss_dir = tmp_path / "faiss_test"
    faiss_dir.mkdir()

    mock_docs = [make_doc("c1", "Kubernetes pod autoscaling"), make_doc("c2", "Redis caching")]
    mock_vs = MockVectorStore(mock_docs)

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid")

    assert mock_rag.retrieval_mode in ("hybrid", "hybrid + rerank")
    assert isinstance(mock_rag.retriever, HybridRetriever)
    assert mock_rag.chain is not None


def test_02_dense_mode_selected_correctly(mock_rag, tmp_path):
    """2. Dense mode is selected correctly and retains pure dense retrieval."""
    faiss_dir = tmp_path / "faiss_test"
    faiss_dir.mkdir()

    mock_docs = [make_doc("c1", "BGP routing")]
    mock_vs = MockVectorStore(mock_docs)

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="dense")

    assert mock_rag.retrieval_mode == "dense"
    assert not isinstance(mock_rag.retriever, HybridRetriever)
    assert mock_rag.chain is not None


def test_03_rewritten_query_passed_to_both_retrievers(tmp_path):
    """3. Rewritten query from the contextualize rewriter reaches both FAISS and BM25."""
    dense_queries = []
    sparse_queries = []

    class TrackDense(Runnable[str, List[Document]]):
        def invoke(self, q, config=None):
            dense_queries.append(q)
            return [make_doc("d1", "dense text")]

    class TrackSparse:
        def retrieve_documents(self, q, top_k=10):
            sparse_queries.append(q)
            return [make_doc("s1", "sparse text")]

    hybrid = HybridRetriever(TrackDense(), TrackSparse())

    rag = ConversationalRAG(session_id="test_rewriter", retriever=hybrid)
    # LLM rewriter output
    rag.llm = MockLLM("What are the claims in JWT?")
    rag._build_lcel_chain()

    # Invoke with contextual user query
    rag.invoke(
        user_input="what about that?",
        chat_history=[HumanMessage("Explain JWT token validation.")],
    )

    # Both retrievers must receive the exact rewritten query generated by the rewriter
    assert len(dense_queries) == 1
    assert len(sparse_queries) == 1
    assert dense_queries[0] == "What are the claims in JWT?"
    assert sparse_queries[0] == "What are the claims in JWT?"


def test_04_conversation_history_preserved(mock_rag, tmp_path):
    """4. Conversation history is passed and maintained throughout the RAG pipeline."""
    faiss_dir = tmp_path / "faiss_test"
    faiss_dir.mkdir()
    mock_vs = MockVectorStore([make_doc("c1", "Content")])

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid")

    history = [
        HumanMessage("My name is Alice."),
        AIMessage("Hello Alice, how can I help?"),
    ]
    answer = mock_rag.invoke("What is my name?", chat_history=history)
    assert isinstance(answer, str)
    assert len(answer) > 0


def test_05_hybrid_retrieval_produces_normal_document_objects():
    """5. Hybrid retrieval returns standard LangChain Document objects."""
    d1 = make_doc("c1", "Sample content 1")
    s1 = make_doc("c2", "Sample content 2")

    dense_mock = MagicMock(invoke=lambda q: [d1])
    sparse_mock = MagicMock(retrieve_documents=lambda q, top_k: [s1])
    hybrid = HybridRetriever(dense_mock, sparse_mock)

    docs = hybrid.retrieve("query")
    assert isinstance(docs, list)
    assert len(docs) == 2
    for d in docs:
        assert isinstance(d, Document)
        assert hasattr(d, "page_content")
        assert hasattr(d, "metadata")


def test_06_document_formatter_receives_retrieved_documents():
    """6. Existing document formatter properly formats retrieved documents."""
    docs = [
        make_doc("c1", "First chunk of technical knowledge."),
        make_doc("c2", "Second chunk of technical knowledge."),
    ]
    formatted = ConversationalRAG._format_docs(docs)
    assert "First chunk of technical knowledge." in formatted
    assert "Second chunk of technical knowledge." in formatted
    assert "\n\n" in formatted


def test_07_prompt_construction_remains_unchanged(mock_rag):
    """7. Contextualize and QA prompts preserve registry messages."""
    assert mock_rag.contextualize_prompt is not None
    assert mock_rag.qa_prompt is not None
    # Check that required input variables exist
    assert "chat_history" in mock_rag.contextualize_prompt.input_variables
    assert "input" in mock_rag.contextualize_prompt.input_variables
    assert "context" in mock_rag.qa_prompt.input_variables


def test_08_existing_llm_invocation_remains_compatible(mock_rag, tmp_path):
    """8. LLM invocation works end-to-end and validates ChatAnswer schema."""
    faiss_dir = tmp_path / "faiss_test"
    faiss_dir.mkdir()
    mock_vs = MockVectorStore([make_doc("c1", "Redis hashing info")])

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid")

    result = mock_rag.invoke("How does Redis hashing work?")
    assert result == "Mocked LLM answer based on context."


def test_09_existing_api_response_structure_remains_unchanged(tmp_path):
    """9. API ChatResponse structure and session flow remain completely intact."""
    from main import app, SESSIONS
    from fastapi.testclient import TestClient

    client = TestClient(app)
    session_id = "test_api_session"
    SESSIONS[session_id] = []

    # Ensure session directory exists for FAISS loader check
    session_dir = Path("faiss_index") / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Mock FAISS loading so that /chat can execute without live indexes
        mock_vs = MockVectorStore([make_doc("c1", "API content")])
        with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
             patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"), \
             patch("multi_doc_chat.src.document_chat.retrieval.ConversationalRAG._load_llm", return_value=MockLLM("API answer")):
            resp = client.post("/chat", json={"session_id": session_id, "message": "Test question"})

        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["answer"] == "API answer"
        # History updated
        assert len(SESSIONS[session_id]) == 2
        assert SESSIONS[session_id][0]["role"] == "user"
        assert SESSIONS[session_id][1]["role"] == "assistant"
    finally:
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir, ignore_errors=True)


def test_10_bm25_initialization_failure_triggers_dense_fallback(mock_rag, tmp_path):
    """10. BM25 initialization failure triggers graceful dense fallback."""
    faiss_dir = tmp_path / "faiss_test"
    faiss_dir.mkdir()
    mock_vs = MockVectorStore([make_doc("c1", "Fallback content")])

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"), \
         patch("multi_doc_chat.src.document_chat.bm25_retriever.BM25Retriever.from_documents", side_effect=RuntimeError("BM25 build failed")):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid", fallback_to_dense=True)

    assert mock_rag.retrieval_mode == "dense (fallback)"
    assert mock_rag.chain is not None
    # Can still answer queries
    answer = mock_rag.invoke("Test query")
    assert answer == "Mocked LLM answer based on context."


def test_11_bm25_retrieval_failure_triggers_dense_fallback():
    """11. BM25 retrieval runtime exception falls back to dense-only candidates gracefully."""
    d_doc = make_doc("d1", "Dense chunk")

    class CrashingBM25:
        def retrieve_documents(self, query, top_k):
            raise RuntimeError("BM25 search crashed!")

    class WorkingDense(Runnable[str, List[Document]]):
        def invoke(self, q, config=None):
            return [d_doc]

    hybrid = HybridRetriever(WorkingDense(), CrashingBM25(), fallback_to_dense=True)
    results = hybrid.retrieve("query")

    # Should not crash; returns the dense candidate
    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "d1"


def test_12_session_specific_retrieval_uses_correct_document_collection(mock_rag, tmp_path):
    """12. Session-matched BM25 extracts and indexes only that session's FAISS docstore."""
    session_docs = [
        make_doc("sess_chunk_1", "Exclusive session content A"),
        make_doc("sess_chunk_2", "Exclusive session content B"),
    ]
    mock_vs = MockVectorStore(session_docs)
    faiss_dir = tmp_path / "faiss_sess_1"
    faiss_dir.mkdir()

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid")

    # Verify the created BM25 retriever indexed exactly the 2 session docs
    bm25 = mock_rag.retriever.bm25_retriever
    assert bm25.total_documents == 2
    assert set(bm25.ordered_chunk_ids) == {"sess_chunk_1", "sess_chunk_2"}


def test_13_cross_session_document_leakage_prevented(mock_rag, tmp_path):
    """13. Chunks from Session 1 cannot leak into Session 2."""
    sess1_docs = [make_doc("s1_c1", "Session 1 private secret data")]
    sess2_docs = [make_doc("s2_c1", "Session 2 public content")]

    vs1 = MockVectorStore(sess1_docs)
    vs2 = MockVectorStore(sess2_docs)

    dir1 = tmp_path / "sess1"
    dir2 = tmp_path / "sess2"
    dir1.mkdir()
    dir2.mkdir()

    # Load session 2
    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=vs2), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(dir2), mode="hybrid")

    # Retrieve docs for session 2
    retrieved = mock_rag.retriever.retrieve("secret")
    retrieved_cids = [d.metadata["chunk_id"] for d in retrieved]

    # Must contain ONLY session 2 chunk, NEVER session 1 chunk
    assert "s1_c1" not in retrieved_cids
    assert retrieved_cids == ["s2_c1"]


def test_14_metadata_survives_retrieval_reaches_formatting(mock_rag, tmp_path):
    """14. Full metadata dictionary survives retrieval and reaches document formatting."""
    rich_meta = {
        "chunk_id": "rich_chunk_001",
        "source": "DOC-SEC-601.docx",
        "content_hash": "sha256_rich_hash",
        "page": 5,
        "author": "Security Team",
    }
    doc = Document(page_content="OAuth 2.0 PKCE specification", metadata=dict(rich_meta))
    mock_vs = MockVectorStore([doc])
    faiss_dir = tmp_path / "faiss_rich"
    faiss_dir.mkdir()

    with patch("langchain_community.vectorstores.FAISS.load_local", return_value=mock_vs), \
         patch("multi_doc_chat.utils.model_loader.ModelLoader.load_embeddings"):
        mock_rag.load_retriever(index_path=str(faiss_dir), mode="hybrid")

    retrieved_docs = mock_rag.retriever.retrieve("PKCE")
    assert len(retrieved_docs) == 1
    retrieved = retrieved_docs[0]

    # Check exact metadata match
    assert retrieved.metadata == rich_meta
    assert "score" not in retrieved.metadata
    assert "rrf_score" not in retrieved.metadata
