import os
import json
import numpy as np
from src.config import get_config
from src.data.symptom_registry import get_symptom_map

# Try importing FAISS; fall back to numpy-based search if unavailable or broken
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def load_gallery():
    """
    Load all per-symptom embeddings at startup.
    Returns dict: symptom_id -> {"embeddings": np.ndarray, "count": int, "display_name": str}
    """
    cfg = get_config()
    embeddings_dir = cfg["data"]["embeddings_dir"]
    symptom_map = get_symptom_map()
    gallery = {}

    for symptom_id, info in symptom_map.items():
        emb_folder = info["embedding_folder"]
        folder_path = os.path.join(embeddings_dir, emb_folder)
        emb_path = os.path.join(folder_path, "embeddings.npy")

        if not os.path.exists(emb_path):
            continue

        embeddings = np.load(emb_path).astype(np.float32)

        gallery[symptom_id] = {
            "embeddings": embeddings,
            "count": embeddings.shape[0],
            "display_name": info["display_name"],
        }

    return gallery


def search_symptom(gallery, symptom_id, query_embedding, top_k=None):
    """
    Search a single symptom's embeddings using cosine similarity.
    Returns (similarities, indices) arrays, sorted descending by similarity.
    """
    if symptom_id not in gallery:
        return None, None

    entry = gallery[symptom_id]
    embeddings = entry["embeddings"]
    count = entry["count"]

    if top_k is None:
        top_k = count
    else:
        top_k = min(top_k, count)

    query = query_embedding.reshape(1, -1).astype(np.float32)

    # Cosine similarity via dot product (works for both L2-normalized and raw vectors)
    query_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = (query_norm @ emb_norms.T).flatten()

    # Get top-k indices
    if top_k < count:
        top_indices = np.argpartition(similarities, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    else:
        top_indices = np.argsort(similarities)[::-1]

    return similarities[top_indices], top_indices
