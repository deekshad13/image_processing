import os
import io
import time
import numpy as np
import cv2
import torch
from PIL import Image
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from typing import List, Optional

from src.config import get_config
from src.data.preprocessing import get_transform
from src.data.symptom_registry import resolve_symptom_id, get_symptom_map
from src.data.gallery_builder import load_gallery, search_symptom
from src.model.encoder import load_model
from src.api.schemas import (
    VerifyResponse, VerifyResult,
    GalleryResponse, GallerySymptom,
    SymptomDetail,
    SettingsResponse, ValidationSettings, SimilaritySettings,
)

router = APIRouter()

# Load model and gallery once at startup
MODEL, MODEL_TYPE = load_model()
GALLERY = load_gallery()
CFG = get_config()
TRANSFORM = get_transform()


def validate_image(contents, content_type):
    val = CFG["validation"]

    if content_type not in val["allowed_formats"]:
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG and PNG allowed.")

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > val["max_file_size_mb"]:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {val['max_file_size_mb']}MB.")

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image. File may be corrupted.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < val["min_brightness"]:
        raise HTTPException(status_code=400, detail="Image is too dark. Please retake in better lighting.")
    if brightness > val["max_brightness"]:
        raise HTTPException(status_code=400, detail="Image is too bright. Please avoid direct sunlight.")

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < val["min_laplacian_var"]:
        raise HTTPException(status_code=400, detail="Image is too blurry. Please retake.")


def _get_recommendation(similarity_pct):
    thresholds = CFG["similarity"]["thresholds"]
    if similarity_pct >= thresholds["high"]:
        return "high", "Take Action — consult an expert immediately"
    elif similarity_pct >= thresholds["medium"]:
        return "medium", "Monitor — observe and recheck in a few days"
    else:
        return "low", "Low Concern — likely not a match"


@router.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_TYPE,
        "symptoms_loaded": len(GALLERY),
    }


@router.get("/gallery", response_model=GalleryResponse)
def get_gallery():
    symptom_map = get_symptom_map()
    symptoms = []
    total = 0
    for symptom_id, info in symptom_map.items():
        count = GALLERY[symptom_id]["count"] if symptom_id in GALLERY else 0
        symptoms.append(GallerySymptom(
            symptom_id=symptom_id,
            display_name=info["display_name"],
            image_count=count,
        ))
        total += count
    return GalleryResponse(symptoms=symptoms, total_images=total)


@router.get("/gallery/{symptom_id}", response_model=SymptomDetail)
def get_symptom_detail(symptom_id: str):
    info = resolve_symptom_id(symptom_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Symptom '{symptom_id}' not found")

    raw_dir = CFG["data"]["raw_dir"]
    folder_path = os.path.join(raw_dir, info["raw_folder"])
    image_count = 0
    if os.path.isdir(folder_path):
        image_count = len([
            f for f in os.listdir(folder_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    return SymptomDetail(
        symptom_id=symptom_id.lower(),
        display_name=info["display_name"],
        description=f"Reference gallery for {info['display_name']} symptom",
        image_count=image_count,
    )


@router.get("/gallery/{symptom_id}/images")
def get_symptom_images(symptom_id: str, limit: int = 20):
    """Return a list of reference image URLs for a symptom."""
    info = resolve_symptom_id(symptom_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Symptom '{symptom_id}' not found")

    raw_dir = CFG["data"]["raw_dir"]
    folder_path = os.path.join(raw_dir, info["raw_folder"])
    if not os.path.isdir(folder_path):
        return {"symptom_id": symptom_id, "images": []}

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:limit]

    from urllib.parse import quote
    images = [
        {"filename": f, "url": f"/images/{quote(info['raw_folder'])}/{quote(f)}"}
        for f in files
    ]
    return {"symptom_id": symptom_id, "display_name": info["display_name"], "images": images}


@router.post("/verify/{symptom_id}", response_model=VerifyResponse)
async def verify_image(symptom_id: str, file: UploadFile = File(...)):
    # Validate symptom exists
    info = resolve_symptom_id(symptom_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Symptom '{symptom_id}' not found")

    symptom_id = symptom_id.lower()
    if symptom_id not in GALLERY:
        raise HTTPException(status_code=404, detail=f"No gallery embeddings for '{symptom_id}'")

    # Validate and process image
    contents = await file.read()
    validate_image(contents, file.content_type)

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        query_emb = MODEL(img_tensor).squeeze().numpy()

    # Search ONLY against the specified symptom's gallery
    gallery_count = GALLERY[symptom_id]["count"]
    top_n = min(CFG["similarity"]["top_n_mean"], gallery_count)
    similarities, _ = search_symptom(GALLERY, symptom_id, query_emb, top_k=top_n)

    # Aggregate: mean of top-N similarities (more robust than single max)
    mean_sim = float(np.mean(similarities[:top_n])) * 100
    mean_sim = round(mean_sim, 1)

    confidence, recommendation = _get_recommendation(mean_sim)
    thresholds = CFG["similarity"]["thresholds"]

    return VerifyResponse(
        status="success",
        result=VerifyResult(
            symptom_id=symptom_id,
            symptom_name=info["display_name"],
            similarity_pct=mean_sim,
            confidence=confidence,
            recommendation=recommendation,
            gallery_size=gallery_count,
            top_n_used=top_n,
        ),
        thresholds={
            "high": f">={thresholds['high']}% — Take Action",
            "medium": f"{thresholds['medium']}-{thresholds['high']-1}% — Monitor",
            "low": f"<{thresholds['medium']}% — Low Concern",
        },
    )


@router.get("/settings", response_model=SettingsResponse)
def get_settings():
    """Get current validation and similarity settings."""
    val = CFG["validation"]
    sim = CFG["similarity"]
    return SettingsResponse(
        validation=ValidationSettings(
            max_file_size_mb=val["max_file_size_mb"],
            min_brightness=val["min_brightness"],
            max_brightness=val["max_brightness"],
            min_laplacian_var=val["min_laplacian_var"],
        ),
        similarity=SimilaritySettings(
            top_n_mean=sim["top_n_mean"],
            threshold_high=sim["thresholds"]["high"],
            threshold_medium=sim["thresholds"]["medium"],
        ),
    )


@router.put("/settings", response_model=SettingsResponse)
def update_settings(settings: SettingsResponse):
    """Update validation and similarity settings at runtime."""
    CFG["validation"]["max_file_size_mb"] = settings.validation.max_file_size_mb
    CFG["validation"]["min_brightness"] = settings.validation.min_brightness
    CFG["validation"]["max_brightness"] = settings.validation.max_brightness
    CFG["validation"]["min_laplacian_var"] = settings.validation.min_laplacian_var
    CFG["similarity"]["top_n_mean"] = settings.similarity.top_n_mean
    CFG["similarity"]["thresholds"]["high"] = settings.similarity.threshold_high
    CFG["similarity"]["thresholds"]["medium"] = settings.similarity.threshold_medium
    return get_settings()


@router.get("/engines")
def list_engines():
    """List all available model engines for comparison."""
    from src.model.engines import list_engines as _list
    return {"engines": _list()}


@router.post("/compare-engines/{symptom_id}")
async def compare_engines(
    symptom_id: str,
    file: UploadFile = File(...),
    engines: Optional[str] = Query(None, description="Comma-separated engine IDs. Omit for all.")
):
    """
    Compare an image against a symptom using multiple model engines side by side.
    Returns results from each engine with similarity scores and timing.
    """
    from src.model.engines import ENGINES, get_embedding_with_engine, load_engine
    from src.data.gallery_builder import search_symptom as _search

    info = resolve_symptom_id(symptom_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Symptom '{symptom_id}' not found")
    symptom_id = symptom_id.lower()

    contents = await file.read()
    validate_image(contents, file.content_type)
    pil_image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Determine which engines to run
    if engines:
        engine_ids = [e.strip() for e in engines.split(",")]
    else:
        engine_ids = list(ENGINES.keys())

    top_n = CFG["similarity"]["top_n_mean"]
    results = []

    for eid in engine_ids:
        if eid not in ENGINES:
            continue

        try:
            start = time.time()

            # Get embedding from this engine
            emb = get_embedding_with_engine(eid, pil_image)
            embed_time = time.time() - start

            # Build temporary gallery for this engine if not the default
            engine_data = load_engine(eid)
            engine_transform = engine_data['transform']
            engine_model = engine_data['model']
            embed_fn = ENGINES[eid]['embed']

            # Compute similarities against symptom's raw images using this engine
            raw_dir = CFG["data"]["raw_dir"]
            folder_path = os.path.join(raw_dir, info["raw_folder"])
            if not os.path.isdir(folder_path):
                continue

            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

            ref_embeddings = []
            for img_file in image_files:
                try:
                    ref_img = Image.open(os.path.join(folder_path, img_file)).convert("RGB")
                    ref_tensor = engine_transform(ref_img).unsqueeze(0)
                    with torch.no_grad():
                        ref_emb = embed_fn(engine_model, ref_tensor)
                    ref_embeddings.append(ref_emb)
                except Exception:
                    continue

            if not ref_embeddings:
                continue

            ref_array = np.array(ref_embeddings, dtype=np.float32)

            # Cosine similarity
            query_norm = emb / (np.linalg.norm(emb) + 1e-8)
            ref_norms = ref_array / (np.linalg.norm(ref_array, axis=1, keepdims=True) + 1e-8)
            similarities = (query_norm @ ref_norms.T).flatten()

            k = min(top_n, len(similarities))
            top_k_sims = np.sort(similarities)[::-1][:k]
            mean_sim = float(np.mean(top_k_sims)) * 100

            total_time = time.time() - start
            confidence, recommendation = _get_recommendation(mean_sim)

            results.append({
                "engine_id": eid,
                "engine_name": ENGINES[eid]["name"],
                "family": ENGINES[eid]["family"],
                "params": ENGINES[eid]["params"],
                "embed_dim": ENGINES[eid]["embed_dim"],
                "similarity_pct": round(mean_sim, 1),
                "confidence": confidence,
                "recommendation": recommendation,
                "top_n_used": k,
                "gallery_size": len(ref_embeddings),
                "inference_ms": round(total_time * 1000),
            })

        except Exception as e:
            results.append({
                "engine_id": eid,
                "engine_name": ENGINES[eid]["name"],
                "error": str(e),
            })

    # Sort by similarity descending
    results.sort(key=lambda r: r.get("similarity_pct", 0), reverse=True)

    return {
        "symptom_id": symptom_id,
        "symptom_name": info["display_name"],
        "results": results,
        "thresholds": {
            "high": CFG["similarity"]["thresholds"]["high"],
            "medium": CFG["similarity"]["thresholds"]["medium"],
        },
    }


@router.post("/compare")
async def compare_deprecated():
    raise HTTPException(
        status_code=410,
        detail="This endpoint has been removed. Use POST /verify/{symptom_id} instead.",
    )
