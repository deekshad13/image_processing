import numpy as np
from typing import List, Dict


def cosine_similarity(embedding_a: np.ndarray,
                      embedding_b: np.ndarray) -> float:
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return round(float(similarity) * 100, 1)


def find_top_matches(query_embedding: np.ndarray,
                     reference_embeddings: np.ndarray,
                     reference_labels: List[str],
                     top_k: int = 5) -> List[Dict]:
    scores = []

    for i, ref_embedding in enumerate(reference_embeddings):
        score = cosine_similarity(query_embedding, ref_embedding)
        scores.append((reference_labels[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for rank, (label, similarity) in enumerate(scores[:top_k], start=1):
        results.append({
            "rank":       rank,
            "symptom":    label,
            "similarity": similarity,
            "action":     _get_action_label(similarity)
        })

    return results


def _get_action_label(similarity: float) -> str:
    if similarity >= 75:
        return "Take Action"
    elif similarity >= 50:
        return "Monitor"
    else:
        return "Low concern"