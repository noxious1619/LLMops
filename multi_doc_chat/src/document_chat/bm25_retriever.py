"""
Standalone BM25 Retrieval and Index Management Module.

Provides:
- BM25IndexBuilder: Builds and persists BM25Okapi index with manifest and chunk mapping.
- BM25Retriever: Standalone BM25 retriever executing lexical search with Top-10 ranking,
  deterministic tie-breaking, and complete metadata preservation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from multi_doc_chat.src.document_chat.bm25_corpus import bm25_tokenize


def sha256_file(path: Path | str) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class BM25IndexBuilder:
    """Builds and serializes a BM25Okapi index from a prepared BM25 corpus."""

    @staticmethod
    def build_and_save(
        corpus_path: Path | str,
        output_dir: Path | str,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> Tuple[Path, Path]:
        """
        Build BM25Okapi from the prepared corpus JSON and save bm25_index.pkl and bm25_manifest.json.

        Args:
            corpus_path: Path to prepared corpus JSON (e.g. data/bm25_corpus/bm25_corpus_70docs_181chunks.json)
            output_dir: Target directory (e.g. data/bm25_index/)
            k1: BM25 term frequency saturation parameter
            b: BM25 document length normalization parameter
            epsilon: BM25 floor parameter for negative IDF

        Returns:
            Tuple of (index_path, manifest_path)
        """
        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            raise FileNotFoundError(f"Prepared corpus not found at {corpus_file}")

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(corpus_file, "r", encoding="utf-8") as f:
            corpus_entries: List[Dict[str, Any]] = json.load(f)

        total_chunks = len(corpus_entries)
        if total_chunks == 0:
            raise ValueError(f"Corpus at {corpus_file} is empty!")

        # 1. Extract pre-computed tokens
        tokenized_corpus: List[List[str]] = []
        ordered_chunk_ids: List[str] = []
        entries_data: List[Dict[str, Any]] = []
        chunk_id_to_index: Dict[str, int] = {}

        for idx, entry in enumerate(corpus_entries):
            cid = entry["chunk_id"]
            if cid in chunk_id_to_index:
                raise ValueError(f"Duplicate chunk_id '{cid}' detected at index {idx}!")

            tokens = entry.get("tokens", [])
            if not tokens:
                raise ValueError(f"Missing or empty tokens for chunk {cid} at index {idx}!")

            tokenized_corpus.append(tokens)
            ordered_chunk_ids.append(cid)
            chunk_id_to_index[cid] = idx

            # Store mapping record
            entries_data.append({
                "index": entry.get("index", idx),
                "chunk_id": cid,
                "docstore_id": entry.get("docstore_id", ""),
                "source": entry.get("source", ""),
                "source_path": entry.get("source_path", ""),
                "page": entry.get("page"),
                "original_text": entry.get("original_text", ""),
                "metadata": entry.get("metadata", {}),
                "token_count": entry.get("token_count", len(tokens)),
                "char_count": entry.get("char_count", len(entry.get("original_text", ""))),
            })

        # 2. Build BM25Okapi instance
        bm25_model = BM25Okapi(
            corpus=tokenized_corpus,
            k1=k1,
            b=b,
            epsilon=epsilon,
        )

        # 3. Serialize index bundle (BM25 model + doc mapping)
        index_bundle = {
            "bm25": bm25_model,
            "entries": entries_data,
            "chunk_id_to_index": chunk_id_to_index,
            "ordered_chunk_ids": ordered_chunk_ids,
        }

        index_path = out_dir / "bm25_index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(index_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

        # 4. Generate manifest
        corpus_sha256 = sha256_file(corpus_file)
        index_sha256 = sha256_file(index_path)

        manifest = {
            "manifest_version": "1.0.0",
            "created_at": datetime.now().isoformat(),
            "algorithm": "BM25Okapi",
            "configuration": {
                "k1": k1,
                "b": b,
                "epsilon": epsilon,
            },
            "total_indexed_chunks": total_chunks,
            "corpus_source": str(corpus_file.as_posix()),
            "corpus_sha256": corpus_sha256,
            "index_sha256": index_sha256,
            "tokenizer_module": "multi_doc_chat.src.document_chat.bm25_corpus",
            "tokenizer_function": "bm25_tokenize",
            "ordered_chunk_ids": ordered_chunk_ids,
        }

        manifest_path = out_dir / "bm25_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        return index_path, manifest_path


class BM25Retriever:
    """
    Standalone BM25 retriever.
    Loads persisted BM25Okapi index and executes deterministic top-K lexical search.
    """

    def __init__(self, index_dir: Path | str = "data/bm25_index"):
        self.index_dir = Path(index_dir)
        self.index_path = self.index_dir / "bm25_index.pkl"
        self.manifest_path = self.index_dir / "bm25_manifest.json"

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"BM25 index file not found at {self.index_path}. "
                "Ensure BM25IndexBuilder.build_and_save() has been run."
            )
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"BM25 manifest not found at {self.manifest_path}."
            )

        self._load_index()

    def _load_index(self):
        """Load and validate the serialized BM25 bundle and manifest."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        with open(self.index_path, "rb") as f:
            bundle = pickle.load(f)

        if not isinstance(bundle, dict) or "bm25" not in bundle or "entries" not in bundle:
            raise ValueError(f"Corrupt or incompatible BM25 index bundle at {self.index_path}")

        self.bm25: BM25Okapi = bundle["bm25"]
        self.entries: List[Dict[str, Any]] = bundle["entries"]
        self.chunk_id_to_index: Dict[str, int] = bundle.get("chunk_id_to_index", {})
        self.ordered_chunk_ids: List[str] = bundle.get("ordered_chunk_ids", [])

        # Validate count consistency
        if len(self.entries) != self.manifest.get("total_indexed_chunks"):
            raise ValueError(
                f"Index count mismatch: bundle has {len(self.entries)} entries, "
                f"manifest specifies {self.manifest.get('total_indexed_chunks')}"
            )

    @property
    def total_documents(self) -> int:
        return len(self.entries)

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Execute standalone BM25 retrieval for a user query.

        Args:
            query: Raw user query string.
            top_k: Number of top results to return (default: 10).

        Returns:
            List of result dicts sorted by BM25 score descending with deterministic tie-breaking.
            Format:
            [
                {
                    "chunk_id": str,
                    "original_text": str,
                    "metadata": dict,
                    "bm25_score": float,
                    "rank": int (1-indexed),
                },
                ...
            ]
        """
        if not query or not query.strip():
            return []

        # 1. Apply exact same tokenizer used during corpus preparation
        query_tokens = bm25_tokenize(query, expand_compounds=True)
        if not query_tokens:
            return []

        # 2. Compute BM25 scores
        raw_scores = self.bm25.get_scores(query_tokens)

        # 3. Create candidate list with deterministic tie-breaking (primary: score desc, secondary: index asc)
        candidates = []
        for idx, score in enumerate(raw_scores):
            # Only consider positive scores (BM25Okapi can produce 0 or negative for non-matches)
            if score > 0.0:
                candidates.append((float(score), idx))

        # Sort: highest score first, ties broken by lower document index
        candidates.sort(key=lambda x: (-x[0], x[1]))

        # 4. Construct Top-K results
        results: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(candidates[:top_k], 1):
            entry = self.entries[idx]
            results.append({
                "chunk_id": entry["chunk_id"],
                "original_text": entry["original_text"],
                "metadata": dict(entry["metadata"]),
                "bm25_score": round(score, 6),
                "rank": rank,
            })

        return results

    def retrieve_documents(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Execute standalone BM25 retrieval and return results as LangChain Document objects.

        Args:
            query: Raw user query string.
            top_k: Number of top results to return (default: 10).

        Returns:
            List of Document objects with original page_content and preserved metadata.
        """
        raw_results = self.retrieve(query, top_k=top_k)
        docs: List[Document] = []
        for r in raw_results:
            docs.append(
                Document(
                    page_content=r["original_text"],
                    metadata=dict(r["metadata"]),
                )
            )
        return docs

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Lookup an original corpus entry by its chunk_id."""
        idx = self.chunk_id_to_index.get(chunk_id)
        if idx is not None and 0 <= idx < len(self.entries):
            return self.entries[idx]
        return None

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> BM25Retriever:
        """
        Build an in-memory BM25Retriever directly from a sequence of Document objects.

        Ensures perfect 1:1 session-matched lexical search over active session chunks
        without writing any temporary files or mutating original metadata.

        Args:
            documents: List of LangChain Document objects.
            k1: BM25 term frequency saturation parameter.
            b: BM25 document length normalization parameter.
            epsilon: BM25 floor parameter for negative IDF.

        Returns:
            Configured, ready-to-query in-memory BM25Retriever instance.
        """
        if not documents:
            raise ValueError("Cannot initialize BM25Retriever from an empty document list")

        tokenized_corpus: List[List[str]] = []
        ordered_chunk_ids: List[str] = []
        entries_data: List[Dict[str, Any]] = []
        chunk_id_to_index: Dict[str, int] = {}

        for idx, doc in enumerate(documents):
            raw_meta = dict(doc.metadata) if hasattr(doc, "metadata") and isinstance(doc.metadata, dict) else {}
            cid = raw_meta.get("chunk_id")
            if not cid or not isinstance(cid, str) or not cid.strip():
                # Derive deterministic stable chunk_id if missing from session document
                content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
                cid = hashlib.sha256(f"{raw_meta.get('source', 'unknown')}_{idx}_{content_hash}".encode("utf-8")).hexdigest()
                raw_meta["chunk_id"] = cid

            cid = cid.strip()
            # Handle rare duplicate chunk_id in docstore defensively by appending index
            if cid in chunk_id_to_index:
                cid = f"{cid}_{idx}"
                raw_meta["chunk_id"] = cid

            tokens = bm25_tokenize(doc.page_content, expand_compounds=True)
            if not tokens:
                tokens = ["<empty>"]

            tokenized_corpus.append(tokens)
            ordered_chunk_ids.append(cid)
            chunk_id_to_index[cid] = idx

            entries_data.append({
                "index": idx,
                "chunk_id": cid,
                "docstore_id": str(idx),
                "source": raw_meta.get("source", ""),
                "source_path": raw_meta.get("source_path", raw_meta.get("source", "")),
                "page": raw_meta.get("page"),
                "original_text": doc.page_content,
                "metadata": raw_meta,
                "token_count": len(tokens),
                "char_count": len(doc.page_content),
            })

        bm25_model = BM25Okapi(tokenized_corpus, k1=k1, b=b, epsilon=epsilon)

        retriever = cls.__new__(cls)
        retriever.index_dir = None
        retriever.index_path = None
        retriever.manifest_path = None
        retriever.bm25 = bm25_model
        retriever.entries = entries_data
        retriever.chunk_id_to_index = chunk_id_to_index
        retriever.ordered_chunk_ids = ordered_chunk_ids
        retriever.manifest = {
            "manifest_version": "1.0.0",
            "algorithm": "BM25Okapi",
            "total_indexed_chunks": len(entries_data),
        }
        return retriever


