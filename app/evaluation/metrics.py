"""
Retrieval and Answer-Quality evaluation metrics for RAG systems.
Implements: Precision@K, Recall@K, Hit Rate, MRR, Context Precision, Faithfulness, and Answer Relevancy.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Set


_WORD_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens."""
    return [t.casefold() for t in _WORD_TOKEN_RE.findall(text or "")]


def is_chunk_relevant(
    chunk: Dict[str, Any],
    expected_snippets: Optional[Sequence[str]] = None,
    relevant_docs: Optional[Sequence[str]] = None,
    relevant_chunk_ids: Optional[Sequence[str]] = None,
) -> bool:
    """
    Determines whether a retrieved chunk is relevant to the ground truth criteria.

    Relevance Criteria:
    1. If `relevant_chunk_ids` are specified, chunk is relevant if its ID matches.
    2. If `expected_snippets` are specified, chunk is relevant if any expected snippet
       is present in the chunk text (via case-insensitive substring or high token overlap >= 0.7).
    3. If only `relevant_docs` are specified (no snippets), matching the source document constitutes relevance.
    4. For negative controls (no docs, no snippets), chunks are never relevant.
    """
    if not chunk:
        return False

    chunk_id = str(chunk.get("chunk_id") or chunk.get("metadata", {}).get("chunk_id") or "")
    if relevant_chunk_ids and chunk_id:
        if chunk_id in relevant_chunk_ids:
            return True

    text = str(chunk.get("text") or "").strip()
    metadata = chunk.get("metadata") or {}
    source = str(chunk.get("source") or metadata.get("source") or metadata.get("filename") or "").strip().lower()

    # Check document match if specified
    doc_matches = True
    if relevant_docs:
        doc_matches = any(
            d.lower() in source or source in d.lower()
            for d in relevant_docs
            if d.strip()
        )
        if not doc_matches:
            return False

    # Check snippets if specified
    if expected_snippets:
        text_lower = text.lower()
        chunk_tokens = set(_tokenize(text))
        for snippet in expected_snippets:
            if not snippet or not snippet.strip():
                continue
            snippet_lower = snippet.strip().lower()
            if snippet_lower in text_lower:
                return True
            # Token set overlap check for minor punctuation/whitespace variations
            snippet_tokens = _tokenize(snippet)
            if snippet_tokens:
                overlap = sum(1 for tok in snippet_tokens if tok in chunk_tokens)
                if (overlap / len(snippet_tokens)) >= 0.7:
                    return True
        return False

    # If relevant_docs matched and no snippets required
    if relevant_docs and doc_matches:
        return True

    return False


def calculate_precision_at_k(
    retrieved_chunks: Sequence[Dict[str, Any]],
    expected_snippets: Optional[Sequence[str]] = None,
    relevant_docs: Optional[Sequence[str]] = None,
    relevant_chunk_ids: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> float:
    """
    Precision@K = (Number of relevant chunks in top K) / K.
    Measures the ratio of retrieved chunks that are actually relevant.
    """
    top_chunks = list(retrieved_chunks)[:k] if k is not None else list(retrieved_chunks)
    if not top_chunks:
        return 0.0

    relevant_count = sum(
        1 for c in top_chunks
        if is_chunk_relevant(c, expected_snippets, relevant_docs, relevant_chunk_ids)
    )
    return relevant_count / len(top_chunks)


def calculate_recall_at_k(
    retrieved_chunks: Sequence[Dict[str, Any]],
    expected_snippets: Optional[Sequence[str]] = None,
    relevant_docs: Optional[Sequence[str]] = None,
    relevant_chunk_ids: Optional[Sequence[str]] = None,
    total_relevant: Optional[int] = None,
    k: Optional[int] = None,
) -> float:
    """
    Recall@K = (Number of relevant chunks in top K) / (Total known relevant items).
    Measures how many of the target information units were successfully retrieved.
    """
    # Total expected relevant units
    expected_count = total_relevant
    if expected_count is None:
        if expected_snippets:
            expected_count = len([s for s in expected_snippets if s.strip()])
        elif relevant_chunk_ids:
            expected_count = len(relevant_chunk_ids)
        elif relevant_docs:
            expected_count = len(relevant_docs)
        else:
            expected_count = 0

    if expected_count <= 0:
        return 0.0

    top_chunks = list(retrieved_chunks)[:k] if k is not None else list(retrieved_chunks)
    if not top_chunks:
        return 0.0

    # Count how many of the expected units are covered by the retrieved chunks
    if expected_snippets:
        combined_text = " ".join(str(c.get("text", "")).lower() for c in top_chunks)
        combined_tokens = set(_tokenize(combined_text))
        found = 0
        for snip in expected_snippets:
            s_lower = snip.strip().lower()
            if s_lower in combined_text:
                found += 1
            else:
                s_tokens = _tokenize(snip)
                if s_tokens and (sum(1 for t in s_tokens if t in combined_tokens) / len(s_tokens)) >= 0.7:
                    found += 1
        return min(1.0, found / expected_count)

    relevant_hits = sum(
        1 for c in top_chunks
        if is_chunk_relevant(c, expected_snippets, relevant_docs, relevant_chunk_ids)
    )
    return min(1.0, relevant_hits / expected_count)


def calculate_hit_rate(
    retrieved_chunks: Sequence[Dict[str, Any]],
    expected_snippets: Optional[Sequence[str]] = None,
    relevant_docs: Optional[Sequence[str]] = None,
    relevant_chunk_ids: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> float:
    """
    Hit Rate@K = 1.0 if at least one relevant chunk is retrieved in top K, else 0.0.
    """
    top_chunks = list(retrieved_chunks)[:k] if k is not None else list(retrieved_chunks)
    for c in top_chunks:
        if is_chunk_relevant(c, expected_snippets, relevant_docs, relevant_chunk_ids):
            return 1.0
    return 0.0


def calculate_reciprocal_rank(
    retrieved_chunks: Sequence[Dict[str, Any]],
    expected_snippets: Optional[Sequence[str]] = None,
    relevant_docs: Optional[Sequence[str]] = None,
    relevant_chunk_ids: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> float:
    """
    Reciprocal Rank = 1 / rank of the first relevant chunk (1-based index).
    Returns 0.0 if no relevant chunk is found in top K.
    """
    top_chunks = list(retrieved_chunks)[:k] if k is not None else list(retrieved_chunks)
    for rank, c in enumerate(top_chunks, start=1):
        if is_chunk_relevant(c, expected_snippets, relevant_docs, relevant_chunk_ids):
            return 1.0 / rank
    return 0.0


def calculate_mrr(reciprocal_ranks: Sequence[float]) -> float:
    """
    Mean Reciprocal Rank (MRR) = Average of reciprocal ranks across all evaluation queries.
    """
    if not reciprocal_ranks:
        return 0.0
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def calculate_context_precision(
    retrieved_chunks: Sequence[Dict[str, Any]],
    expected_snippets: Optional[Sequence[str]] = None,
    relevant_docs: Optional[Sequence[str]] = None,
    relevant_chunk_ids: Optional[Sequence[str]] = None,
    k: Optional[int] = None,
) -> float:
    """
    Context Precision@K evaluates whether relevant chunks are ranked higher in the context window.
    Sum of (Precision@i * Relevance(i)) / (Total relevant in top K).
    """
    top_chunks = list(retrieved_chunks)[:k] if k is not None else list(retrieved_chunks)
    if not top_chunks:
        return 0.0

    precisions: List[float] = []
    relevant_count = 0
    cumulative_score = 0.0

    for i, c in enumerate(top_chunks, start=1):
        rel = 1 if is_chunk_relevant(c, expected_snippets, relevant_docs, relevant_chunk_ids) else 0
        if rel:
            relevant_count += 1
            precision_at_i = relevant_count / i
            cumulative_score += precision_at_i

    if relevant_count == 0:
        return 0.0

    return cumulative_score / relevant_count


def calculate_faithfulness(
    answer: str,
    context_chunks: Sequence[Dict[str, Any]],
    refusal_message: str = "I couldn't find sufficient evidence in the provided documents to answer that question.",
) -> float:
    """
    Measures the degree to which claims in the generated answer are grounded in the retrieved context.
    Evaluates sentence-by-sentence containment and lexical overlap.
    Returns a score between 0.0 and 1.0.
    """
    if not answer or not answer.strip():
        return 0.0

    # If the answer is an intentional refusal when context was insufficient, it is 100% faithful
    clean_ans = answer.strip()
    if refusal_message.lower() in clean_ans.lower():
        return 1.0

    context_text = " ".join(str(c.get("text", "")) for c in context_chunks).strip()
    if not context_text:
        return 0.0

    context_tokens = set(_tokenize(context_text))

    # Split answer into clean sentences/clauses
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", clean_ans) if len(s.strip()) > 8]
    if not sentences:
        # Fallback to entire answer token overlap
        ans_tokens = _tokenize(clean_ans)
        if not ans_tokens:
            return 0.0
        supported = sum(1 for t in ans_tokens if t in context_tokens)
        return min(1.0, supported / len(ans_tokens))

    supported_sentences = 0
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "to", "for", "of", "with"}

    for sentence in sentences:
        s_tokens = [t for t in _tokenize(sentence) if t not in stopwords and len(t) > 2]
        if not s_tokens:
            supported_sentences += 1
            continue
        overlap = sum(1 for t in s_tokens if t in context_tokens)
        if (overlap / len(s_tokens)) >= 0.5:
            supported_sentences += 1

    return supported_sentences / len(sentences)


def calculate_answer_relevancy(
    answer: str,
    question: str,
    expected_answer: Optional[str] = None,
) -> float:
    """
    Measures lexical and semantic alignment between the answer and the question/ground truth.
    Uses token overlap and F1 scoring (harmonic mean of precision and recall against expected answer).
    Returns a score between 0.0 and 1.0.
    """
    if not answer or not answer.strip():
        return 0.0

    ans_tokens = set(_tokenize(answer))
    q_tokens = set(_tokenize(question))

    if not ans_tokens:
        return 0.0

    # If expected answer is available, compute token-level F1 score
    if expected_answer and expected_answer.strip():
        target_tokens = set(_tokenize(expected_answer))
        if target_tokens:
            common = ans_tokens.intersection(target_tokens)
            precision = len(common) / len(ans_tokens)
            recall = len(common) / len(target_tokens)
            if precision + recall > 0:
                f1 = (2 * precision * recall) / (precision + recall)
                return min(1.0, f1)

    # Fallback to question token alignment
    stopwords = {"what", "who", "when", "where", "which", "how", "why", "does", "is", "are", "the", "a", "an"}
    informative_q = {t for t in q_tokens if t not in stopwords and len(t) > 2}
    if not informative_q:
        return 0.8  # Default baseline for non-informative questions

    overlap = len(ans_tokens.intersection(informative_q))
    return min(1.0, overlap / len(informative_q))
