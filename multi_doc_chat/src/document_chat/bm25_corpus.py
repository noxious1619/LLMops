"""
BM25 Corpus Preparation and Tokenization Module.

Provides:
- Conservative, technical-preserving tokenizer (bm25_tokenize)
- BM25CorpusEntry data structure
- BM25CorpusBuilder for extracting and preparing BM25-ready representations
  directly from the canonical FAISS index.
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def clean_text_for_tokenization(text: str) -> str:
    """Normalize typographical quotes and dashes to standard ASCII equivalents."""
    if not text:
        return ""
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("—", " - ").replace("–", " - ")
    return text


def bm25_tokenize(text: str, expand_compounds: bool = True) -> List[str]:
    """
    Conservative, technical-preserving tokenizer designed for BM25 retrieval.

    Guarantees:
    - Preserves technical acronyms & terms: eBGP, iBGP, PostgreSQL, JWT, OAuth 2.0, HPA, CRC16
    - Preserves programming language markers: C++, C#
    - Preserves decimal and version identifiers: 2.0, v1.2, 1.0.0
    - Preserves hyphenated, dotted, and underscored identifiers: brake-by-wire, sha-256, doc-net-802
    - Decomposes hyphenated/underscored compounds when expand_compounds=True to allow both
      exact compound matching ("brake-by-wire") and constituent sub-term matching ("brake", "wire").
    - Symmetric: Used identically for indexing documents and tokenizing incoming user queries.
    """
    if not text:
        return []

    cleaned = clean_text_for_tokenization(text)

    # Regex breakdown:
    # 1. c\+\+|c\#: Explicitly capture C++ and C#
    # 2. [a-zA-Z0-9]+(?:[-._][a-zA-Z0-9]+)*: Capture alphanumeric sequences with internal
    #    hyphens, periods, or underscores without splitting words like '2.0', 'sha-256', 'doc-net-802'
    pattern = re.compile(r"c\+\+|c\#|[a-zA-Z0-9]+(?:[-._][a-zA-Z0-9]+)*", re.IGNORECASE)
    raw_tokens = pattern.findall(cleaned.lower())

    if not expand_compounds:
        return raw_tokens

    tokens: List[str] = []
    for tok in raw_tokens:
        tokens.append(tok)
        if "-" in tok or "_" in tok:
            sub_parts = re.split(r"[-_]", tok)
            for part in sub_parts:
                # Add sub-token if it is non-empty, distinct, and not a single letter/digit
                if part and part != tok and len(part) > 1 and not part.isdigit():
                    tokens.append(part)
    return tokens


@dataclass
class BM25CorpusEntry:
    """Represents a single document chunk prepared for BM25 indexing."""
    index: int
    chunk_id: str
    docstore_id: str
    source: str
    source_path: str
    page: Optional[int]
    original_text: str
    tokens: List[str]
    token_count: int
    char_count: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BM25CorpusBuilder:
    """Extracts chunks from a FAISS index docstore and prepares a BM25 corpus."""

    def __init__(self, index_dir: Path | str):
        self.index_dir = Path(index_dir)
        self.pkl_path = self.index_dir / "index.pkl"
        if not self.pkl_path.exists():
            raise FileNotFoundError(f"FAISS index.pkl not found at {self.pkl_path}")

    def load_docstore(self) -> Tuple[Any, Dict[int, str]]:
        with open(self.pkl_path, "rb") as f:
            docstore, index_to_id = pickle.load(f)
        return docstore, index_to_id

    def build_corpus(self) -> List[BM25CorpusEntry]:
        docstore, index_to_id = self.load_docstore()
        total_vectors = len(index_to_id)

        corpus: List[BM25CorpusEntry] = []

        for idx in range(total_vectors):
            docstore_id = index_to_id[idx]
            doc = docstore.search(docstore_id)
            if doc is None:
                raise ValueError(f"Missing docstore document for index {idx} (ID: {docstore_id})")

            meta = dict(doc.metadata or {})
            chunk_id = meta.get("chunk_id", "")
            if not chunk_id:
                raise ValueError(f"Missing chunk_id in metadata for index {idx}")

            raw_source = meta.get("source", "")
            source_filename = Path(raw_source).name if raw_source else "unknown"
            page = meta.get("page")
            if isinstance(page, str) and page.isdigit():
                page = int(page)
            elif not isinstance(page, int):
                page = None

            original_text = doc.page_content or ""
            if not original_text.strip():
                raise ValueError(f"Empty text in chunk {chunk_id} at index {idx}")

            tokens = bm25_tokenize(original_text, expand_compounds=True)

            entry = BM25CorpusEntry(
                index=idx,
                chunk_id=chunk_id,
                docstore_id=docstore_id,
                source=source_filename,
                source_path=raw_source,
                page=page,
                original_text=original_text,
                tokens=tokens,
                token_count=len(tokens),
                char_count=len(original_text),
                metadata=meta,
            )
            corpus.append(entry)

        return corpus

    def save_corpus(self, output_path: Path | str, corpus: List[BM25CorpusEntry]) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        data = [e.to_dict() for e in corpus]
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return out_path

