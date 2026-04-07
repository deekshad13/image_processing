import numpy as np
from typing import List, Dict
from src.config import get_config


def cosine_similarity(embedding_a, embedding_b):
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)
    return round(float(similarity) * 100, 1)


def get_recommendation(similarity_pct):
    cfg = get_config()
    thresholds = cfg["similarity"]["thresholds"]
    if similarity_pct >= thresholds["high"]:
        return "high", "Take Action"
    elif similarity_pct >= thresholds["medium"]:
        return "medium", "Monitor"
    else:
        return "low", "Low Concern"


def find_top_matches(query_embedding, reference_embeddings, reference_labels, top_k=5):
    scores = []

    for i, ref_embedding in enumerate(reference_embeddings):
        score = cosine_similarity(query_embedding, ref_embedding)
        scores.append((reference_labels[i], score))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for rank, (label, similarity) in enumerate(scores[:top_k], start=1):
        confidence, action = get_recommendation(similarity)
        results.append({
            "rank": rank,
            "symptom": label,
            "similarity": similarity,
            "action": action,
        })

    return results
