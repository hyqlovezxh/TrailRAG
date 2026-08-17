"""Sentence splitting and semantic clustering helpers.

Ported from YUSU ``yuxi.knowledge.chunking.ragflow_like.utils.semantic_utils``,
with NLTK (punkt_tab download) and scikit-learn replaced by a self-contained
numpy implementation so the demo needs no model downloads and no heavy deps:
- English/mixed sentence splitting uses a regex splitter instead of
  ``nltk.tokenize.sent_tokenize``.
- Agglomerative clustering (average linkage / UPGMA over cosine distance) is
  implemented directly; ``silhouette_score`` is computed from the distance
  matrix.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import numpy as np

# Matches sentence-ending punctuation followed by whitespace and an uppercase
# letter / quote / paren, typical of English text. Abbreviations may produce
# extra splits; acceptable for demo-quality chunking.
_EN_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences_chinese(text: str) -> list[str]:
    """Split Chinese text into sentences with a regex.

    Matches 。！？ as split points; lookahead/lookbehind keep a trailing quote
    (”’") attached to the current sentence.
    """
    pattern = r'(?<=[。！？][”’"])|(?<=[。！？])(?![”’"])'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]


def split_sentences_english(text: str) -> list[str]:
    """Split English/mixed text into sentences with a regex (NLTK-free)."""
    parts = _EN_SENTENCE_BOUNDARY_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def split_mixed_sentences(text: str) -> list[str]:
    """Split mixed Chinese/English text, dispatching per physical paragraph:
    - paragraphs containing ASCII letters go through the English splitter;
    - pure Chinese paragraphs use split_sentences_chinese;
    - fallback: force-split on Chinese punctuation.
    """
    chunks = re.split(r"(\n+)", text)
    sentences = []

    for ch in chunks:
        if not ch.strip():
            continue
        if re.search(r"[A-Za-z]", ch):
            parts = split_sentences_english(ch)
            sentences.extend([p.strip() for p in parts if p.strip()])
        else:
            sents = split_sentences_chinese(ch)
            if sents:
                sentences.extend([s.strip() for s in sents if s.strip()])
            else:
                parts = re.split(r"(?<=[。！？])", ch)
                sentences.extend([p.strip() for p in parts if p.strip()])
    return sentences


def cosine_distance_matrix(embeddings: Any) -> np.ndarray:
    """Cosine distance matrix (1 - cosine similarity) over a 2D embedding array."""
    vecs = np.asarray(embeddings, dtype=np.float64)
    if vecs.ndim != 2:
        raise ValueError(f"embeddings must be 2D, got shape {vecs.shape}")
    norms = np.linalg.norm(vecs, axis=1)
    norms[norms == 0] = 1.0
    normed = vecs / norms[:, None]
    sim = normed @ normed.T
    np.fill_diagonal(sim, 1.0)
    return np.clip(1.0 - sim, 0.0, 2.0)


def agglomerative_cluster_labels(
    embeddings: Any, n_clusters: int, linkage: str = "average", metric: str = "cosine"
) -> np.ndarray:
    """Average-linkage (UPGMA) agglomerative clustering over cosine distance.

    Replacement for ``sklearn.cluster.AgglomerativeClustering(...).fit_predict``
    with ``metric="cosine", linkage="average"``. Cluster ids are arbitrary.
    """
    if linkage != "average" or metric != "cosine":
        raise ValueError(f"Unsupported linkage/metric: {linkage}/{metric}")

    n = len(embeddings)
    if n_clusters <= 1:
        return np.zeros(n, dtype=np.int64)
    if n <= n_clusters:
        return np.arange(n, dtype=np.int64)

    dist = cosine_distance_matrix(embeddings)
    np.fill_diagonal(dist, np.inf)

    sizes: dict[int, int] = {i: 1 for i in range(n)}
    members: dict[int, list[int]] = {i: [i] for i in range(n)}
    active = list(range(n))

    while len(active) > n_clusters:
        sub = dist[np.ix_(active, active)]
        np.fill_diagonal(sub, np.inf)
        flat_idx = int(np.argmin(sub))
        row, col = np.unravel_index(flat_idx, sub.shape)
        u, v = active[row], active[col]

        members[u].extend(members[v])
        sizes[u] += sizes[v]
        members.pop(v)
        active.remove(v)

        old_u_size = sizes[u] - sizes[v]
        for w in active:
            if w == u:
                continue
            du = dist[u, w]
            dv = dist[v, w]
            updated = (old_u_size * du + sizes[v] * dv) / (old_u_size + sizes[v])
            dist[u, w] = updated
            dist[w, u] = updated
        dist[v, :] = np.inf
        dist[:, v] = np.inf

    labels = np.zeros(n, dtype=np.int64)
    for label, members_list in enumerate(members.values()):
        for idx in members_list:
            labels[idx] = label
    return labels


def silhouette_score(embeddings: Any, labels: Any) -> float:
    """Mean silhouette coefficient computed from the cosine distance matrix."""
    dist = cosine_distance_matrix(embeddings)
    labels_arr = np.asarray(labels, dtype=np.int64)
    n = len(dist)
    if n == 0:
        return 0.0

    unique_labels = np.unique(labels_arr)
    if len(unique_labels) < 2:
        return 0.0

    cluster_ids = {label: np.where(labels_arr == label)[0] for label in unique_labels}
    for cluster in cluster_ids.values():
        if len(cluster) < 2:
            cluster_ids = {label: idx for label, idx in cluster_ids.items() if len(idx) >= 2}
            if len(cluster_ids) < 2:
                return 0.0
            break

    total = 0.0
    count = 0
    for i in range(n):
        own_label = labels_arr[i]
        own_idx = cluster_ids.get(own_label)
        if own_idx is None or len(own_idx) < 2:
            continue
        a = float(np.mean(dist[i, own_idx[own_idx != i]]))
        b = np.inf
        for label, idx in cluster_ids.items():
            if label == own_label:
                continue
            b = min(b, float(np.mean(dist[i, idx])))
        if b == np.inf:
            continue
        denom = max(a, b)
        if denom > 0:
            total += (b - a) / denom
            count += 1

    return total / count if count else 0.0


def find_best_num_clusters(embeddings: Any, min_clusters: int = 2, max_clusters: int = 10) -> int:
    """Choose the best cluster count via silhouette score (1 if impossible)."""
    n = len(embeddings)
    if n <= min_clusters:
        return n

    best_score = -1.0
    best_k = min_clusters

    limit_k = min(max_clusters, n)
    for k in range(min_clusters, limit_k + 1):
        labels = agglomerative_cluster_labels(embeddings, k)
        if len(set(labels.tolist())) <= 1:
            continue
        score = silhouette_score(embeddings, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def semantic_chunking_with_auto_clusters(
    text: str,
    embed_fn: Callable[[list[str]], Any] | None,
    token_count_fn: Callable[[str], int],
    max_chunk_size: int = 1024,
) -> list[str]:
    """Semantically chunk text, auto-selecting the cluster count.

    - sentences are split by language (English/mixed via regex, Chinese via
      punctuation);
    - sentences are embedded, then clustered (average linkage, cosine);
    - chunks are cut at cluster-label changes or length limits, preserving
      original order;
    - without an embed_fn this degrades to simple length-based merging.
    """
    sentences = split_mixed_sentences(text)
    if len(sentences) < 2:
        return [text.strip()]

    sentence_token_counts = [token_count_fn(s) for s in sentences]
    total_tokens = sum(sentence_token_counts)

    if embed_fn is None or total_tokens <= max_chunk_size:
        chunks = []
        current_chunk = ""
        current_chunk_tokens = 0
        for s, cnt in zip(sentences, sentence_token_counts):
            if current_chunk_tokens + cnt > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = s
                current_chunk_tokens = cnt
            else:
                current_chunk += s
                current_chunk_tokens += cnt
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    embeddings = embed_fn(sentences)

    best_k = (total_tokens + max_chunk_size - 1) // max_chunk_size
    best_k = min(best_k, len(sentences))

    labels = agglomerative_cluster_labels(embeddings, best_k)

    chunks = []
    current_chunk = ""
    current_chunk_tokens = 0
    current_label = labels[0]

    for sentence, label, token_count in zip(sentences, labels, sentence_token_counts):
        if label != current_label or current_chunk_tokens + token_count > max_chunk_size:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_chunk_tokens = token_count
            current_label = label
        else:
            current_chunk += sentence
            current_chunk_tokens += token_count

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks