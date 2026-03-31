from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from src.api.schemas import CompareResponse, MatchResult, SymptomDetail
from src.model.encoder import load_dinov2
from src.model.similarity import cosine_similarity
from typing import List
from src.config import (
    MAX_FILE_SIZE_MB, MIN_BRIGHTNESS, MAX_BRIGHTNESS, MIN_LAPLACIAN
)
import os, io, numpy as np, random
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2

router = APIRouter()

DATA_RAW_DIR        = "data/raw"
DATA_EMBEDDINGS_DIR = "data/embeddings"
PROTOTYPES_PATH     = os.path.join(DATA_EMBEDDINGS_DIR, "prototypes.pt")

MODEL = load_dinov2()

def load_prototypes():
    if not os.path.exists(PROTOTYPES_PATH):
        raise RuntimeError(
            f"prototypes.pt not found at {PROTOTYPES_PATH}. "
            "Run: python scripts/compute_prototypes.py"
        )
    data        = torch.load(PROTOTYPES_PATH, map_location="cpu")
    proto_tensor = data["prototypes"]    # [10, 128]
    class_names  = data["class_names"]   # ["Burnt_appearance", ...]
    
    # build { symptom_name: np.ndarray }
    prototypes = {
        class_names[i]: proto_tensor[i].numpy()
        for i in range(len(class_names))
    }
    return prototypes

PROTOTYPES = load_prototypes()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

ALLOWED_FORMATS = {"image/jpeg", "image/png", "image/jpg"}

def validate_image(contents: bytes, content_type: str):
    if content_type not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG and PNG allowed.")
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image. File may be corrupted.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < MIN_BRIGHTNESS:
        raise HTTPException(status_code=400, detail="Image is too dark. Please retake in better lighting.")
    if brightness > MAX_BRIGHTNESS:
        raise HTTPException(status_code=400, detail="Image is too bright. Please avoid direct sunlight.")
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < MIN_LAPLACIAN:
        raise HTTPException(status_code=400, detail="Image is too blurry. Please retake with a steady hand.")


def get_reference_images(symptom_name: str, n: int = 4) -> List[str]:
    """Pick up to n reference image paths from data/raw for the matched symptom."""
    matched = next(
        (s for s in os.listdir(DATA_RAW_DIR)
         if s.lower().replace(" ", "_") == symptom_name.lower().replace(" ", "_")),
        None
    )
    if matched is None:
        return []
    folder = os.path.join(DATA_RAW_DIR, matched)
    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    random.shuffle(images)
    return [f"/images/{matched}/{f}" for f in images[:n]]


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/gallery")
def get_gallery():
    if not os.path.exists(DATA_RAW_DIR):
        raise HTTPException(
            status_code=500,
            detail=f"data/raw not found at: {os.path.abspath(DATA_RAW_DIR)}"
        )
    symptoms = [
        s for s in os.listdir(DATA_RAW_DIR)
        if os.path.isdir(os.path.join(DATA_RAW_DIR, s))
    ]
    return {"symptoms": symptoms}


@router.get("/gallery/{symptom_id}/images")
def get_symptom_images(symptom_id: str):
    matched = next(
        (s for s in os.listdir(DATA_RAW_DIR)
         if s.lower().replace(" ", "_") == symptom_id.lower().replace(" ", "_")),
        None
    )
    if matched is None:
        raise HTTPException(status_code=404, detail="Symptom not found")
    images = [
        f for f in os.listdir(os.path.join(DATA_RAW_DIR, matched))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:5]
    return {"images": [f"/images/{matched}/{f}" for f in images]}


@router.get("/gallery/{symptom_id}/first-image")
def get_first_image(symptom_id: str):
    matched = next(
        (s for s in os.listdir(DATA_RAW_DIR)
         if s.lower().replace(" ", "_") == symptom_id.lower().replace(" ", "_")),
        None
    )
    if matched is None:
        raise HTTPException(status_code=404, detail="Symptom not found")
    images = [
        f for f in os.listdir(os.path.join(DATA_RAW_DIR, matched))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not images:
        raise HTTPException(status_code=404, detail="No images found")
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/images/{matched}/{images[0]}")


@router.get("/gallery/{symptom_id}")
def get_symptom_detail(symptom_id: str):
    matched = next(
        (s for s in os.listdir(DATA_RAW_DIR)
         if s.lower().replace(" ", "_") == symptom_id.lower().replace(" ", "_")),
        None
    )
    if matched is None:
        raise HTTPException(status_code=404, detail=f"Symptom '{symptom_id}' not found")
    image_count = len([
        f for f in os.listdir(os.path.join(DATA_RAW_DIR, matched))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return SymptomDetail(
        name=matched,
        description=f"Reference gallery for {matched.replace('_', ' ')} symptom",
        severity="high" if image_count < 20 else "medium",
        action=f"Found {image_count} reference images in gallery"
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_image(file: UploadFile = File(...), symptom: str = Form(None)):
    contents = await file.read()
    validate_image(contents, file.content_type)

    # Embed the query image
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        query_emb = MODEL(img_tensor).squeeze().numpy()

    # Compare query embedding against each prototype (one vector per symptom)
    if symptom:
        proto_source = {
            k: v for k, v in PROTOTYPES.items()
            if k.lower().replace(" ", "_") == symptom.lower().replace(" ", "_")
        }
        if not proto_source:
            proto_source = PROTOTYPES
    else:
        proto_source = PROTOTYPES

    scores = []
    for sym, proto_emb in proto_source.items():
        sim = cosine_similarity(query_emb, proto_emb) * 100  # scale to 0-100
        scores.append((sim, sym))

    scores.sort(reverse=True)

    best_sim, best_symptom = scores[0]
    best_sim   = round(float(best_sim), 1)
    second_sim = round(float(scores[1][0]), 1) if len(scores) > 1 else 0
    gap        = best_sim - second_sim

    # --- Thresholds (same semantics as before) ---
    if symptom and best_sim < 60:
        return CompareResponse(
            status="wrong_symptom",
            plant_part_detected="unknown",
            matches=[],
            thresholds={
                "high_match": "80%+ — Likely this condition",
                "medium_match": "60-79% — Possible, monitor closely",
                "low_match": "Below 60% — Unlikely match"
            }
        )

    if not symptom and best_sim < 70 and gap < 5:
        return CompareResponse(
            status="unknown",
            plant_part_detected="unknown",
            matches=[],
            thresholds={
                "high_match": "80%+ — Likely this condition",
                "medium_match": "60-79% — Possible, monitor closely",
                "low_match": "Below 60% — Unlikely match"
            }
        )

    # --- Severity + action ---
    if best_symptom.lower().replace(" ", "_") == "leaf_healthy":
        severity = "low"
        action   = "No Action Needed — crop appears healthy"
    elif best_sim >= 80:
        severity = "high"
        action   = "Take Action — consult an expert immediately"
    elif best_sim >= 60:
        severity = "medium"
        action   = "Monitor — observe and recheck in a few days"
    else:
        severity = "low"
        action   = "Low Concern — likely not a match"

    # Pull reference images from data/raw for the matched symptom
    ref_images = get_reference_images(best_symptom, n=4)

    matches = [
        MatchResult(
            symptom=best_symptom,
            similarity_pct=best_sim,
            severity=severity,
            description=f"Reference image {i + 1} for {best_symptom.replace('_', ' ')}",
            action=action,
            reference_image=ref_image
        )
        for i, ref_image in enumerate(ref_images)
    ]

    return CompareResponse(
        status="success",
        plant_part_detected="unknown",
        matches=matches,
        thresholds={
            "high_match": "80%+ — Likely this condition",
            "medium_match": "60-79% — Possible, monitor closely",
            "low_match": "Below 60% — Unlikely match"
        }
    )