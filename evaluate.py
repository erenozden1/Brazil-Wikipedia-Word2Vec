# evaluate.py
# Responsibility: qualitative evaluation of the trained word vectors.

import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def nearest_neighbours(word: str,
                       W_in: np.ndarray,
                       word2idx: dict,
                       idx2word: list,
                       top_n: int = 5) -> list[tuple[str, float]]:
    if word not in word2idx:
        print(f"  '{word}' not in vocabulary")
        return []

    query_vec  = W_in[word2idx[word]]

    norms      = np.linalg.norm(W_in, axis=1, keepdims=True)
    norms      = np.where(norms == 0, 1e-10, norms)
    W_norm     = W_in / norms

    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    sims       = W_norm.dot(query_norm)# cosine similarity to every word

    top_indices = np.argsort(sims)[-(top_n + 1):][::-1]

    results = []
    for idx in top_indices:
        if idx2word[idx] != word:
            results.append((idx2word[idx], float(sims[idx])))
        if len(results) == top_n:
            break
    return results


def word_analogy(positive: list[str],
                 negative: list[str],
                 W_in: np.ndarray,
                 word2idx: dict,
                 idx2word: list,
                 top_n: int = 3) -> list[tuple[str, float]]:
    query           = np.zeros(W_in.shape[1], dtype=np.float64)
    all_input_words = positive + negative

    for w in positive:
        if w in word2idx:
            query += W_in[word2idx[w]]
    for w in negative:
        if w in word2idx:
            query -= W_in[word2idx[w]]

    norm = np.linalg.norm(query)
    if norm == 0:
        return []
    query /= norm

    norms   = np.linalg.norm(W_in, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1e-10, norms)
    W_norm  = W_in / norms
    sims    = W_norm.dot(query)

    top_indices = np.argsort(sims)[::-1]
    results = []
    for idx in top_indices:
        candidate = idx2word[idx]
        if candidate not in all_input_words:
            results.append((candidate, float(sims[idx])))
        if len(results) == top_n:
            break
    return results
