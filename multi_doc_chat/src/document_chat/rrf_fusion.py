"""
Reciprocal Rank Fusion (RRF) Module.

Combines ranked result lists from multiple retrieval systems (e.g. dense FAISS
and sparse BM25) into a unified, deduplicated, and deterministically sorted ranking.

Mathematical Formula:
    RRF(d) = sum_{r in R} 1 / (k + rank_r(d))

Where:
    - d is a document chunk
    - R is the set of ranked result lists
    - k is the RRF smoothing constant (default: 60)
    - rank_r(d) is the 1-based rank of document d in list r
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set
from langchain_core.documents import Document


class RRFResult(NamedTuple):
    """
    Structured representation of a fused retrieval result.

    Attributes:
        document: The original, immutable Document object.
        score: The calculated RRF score.
        rank: The 1-based final rank position.
    """
    document: Document
    score: float
    rank: int


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = 60,
) -> List[RRFResult]:
    """
    Execute Reciprocal Rank Fusion on multiple ranked lists of Document objects.

    Rules:
    - Deduplicates chunks using stable `document.metadata["chunk_id"]`.
    - Preserves the original Document object and metadata without modification.
    - Handles repeated chunk IDs within a single list defensively (counts first occurrence only).
    - Preserves the first valid Document object encountered for any given chunk_id.
    - Explicitly validates that each document has a non-empty, string `chunk_id`.
    - Sorts fused documents by score descending, with deterministic tie-breaking
      on chunk_id ascending.

    Args:
        ranked_lists: Sequence of ranked lists containing Document objects.
        k: Smoothing constant (default: 60). Must be greater than 0.

    Returns:
        List of RRFResult namedtuples sorted by score descending.

    Raises:
        ValueError: If k <= 0, or if any document is missing a valid 'chunk_id' in metadata.
        TypeError: If an element in any ranked list is not a Document instance.
    """
    if k <= 0:
        raise ValueError(f"RRF smoothing parameter k must be positive, got {k}")

    if not ranked_lists:
        return []

    scores: Dict[str, float] = {}
    doc_store: Dict[str, Document] = {}

    for list_idx, ranked_list in enumerate(ranked_lists):
        seen_in_current_list: Set[str] = set()

        for rank_0, item in enumerate(ranked_list):
            if not isinstance(item, Document):
                raise TypeError(
                    f"Expected Document object at list index {list_idx}, rank {rank_0 + 1}; "
                    f"got {type(item).__name__}"
                )

            # Metadata validation
            if not hasattr(item, "metadata") or not isinstance(item.metadata, dict):
                raise ValueError(
                    f"Document at list index {list_idx}, rank {rank_0 + 1} "
                    f"is missing a valid metadata dictionary: {item}"
                )

            chunk_id = item.metadata.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str) or not chunk_id.strip():
                raise ValueError(
                    f"Document at list index {list_idx}, rank {rank_0 + 1} "
                    f"has missing or invalid 'chunk_id' in metadata: {item.metadata}"
                )

            chunk_id = chunk_id.strip()

            # Preserve the first valid Document object encountered
            if chunk_id not in doc_store:
                doc_store[chunk_id] = item

            # Deduplicate within the same list: count only the best (first) rank contribution
            if chunk_id in seen_in_current_list:
                continue
            seen_in_current_list.add(chunk_id)

            rank_1 = rank_0 + 1
            reciprocal_score = 1.0 / (k + rank_1)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + reciprocal_score

    if not scores:
        return []

    # Deterministic sorting:
    # Primary key: score descending (-scores[cid])
    # Secondary key: chunk_id ascending (cid)
    sorted_chunk_ids = sorted(
        scores.keys(),
        key=lambda cid: (-scores[cid], cid)
    )

    results: List[RRFResult] = []
    for final_rank, cid in enumerate(sorted_chunk_ids, start=1):
        results.append(
            RRFResult(
                document=doc_store[cid],
                score=scores[cid],
                rank=final_rank,
            )
        )

    return results


def rrf_fuse_documents(
    ranked_lists: Sequence[Sequence[Document]],
    k: int = 60,
    top_n: Optional[int] = None,
) -> List[Document]:
    """
    Convenience wrapper returning pure Document objects after RRF fusion.

    Args:
        ranked_lists: Sequence of ranked lists containing Document objects.
        k: Smoothing constant (default: 60).
        top_n: Optional limit on the number of documents to return.

    Returns:
        List of Document objects sorted by fused RRF rank.
    """
    fused_results = reciprocal_rank_fusion(ranked_lists, k=k)
    if top_n is not None and top_n >= 0:
        fused_results = fused_results[:top_n]
    return [r.document for r in fused_results]

