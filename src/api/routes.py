from fastapi import APIRouter, UploadFile, File, HTTPException
from src.api.schemas import CompareResponse, MatchResult, SymptomDetail
from src.model.encoder import load_dinov2, get_embedding
from src.model.similarity import cosine_similarity
import os, io, numpy as np, random
from PIL import Image
import torchvision.transforms as transforms
import torch
import cv2

router = APIRouter()

MODEL = load_dinov2()
EMBEDDINGS_DIR = "data/embeddings"

def load_gallery():
    gallery = {}
    for symptom in os.listdir(EMBEDDINGS_DIR):
        path = os.path.join(EMBEDDINGS_DIR, symptom)
        if not os.path.isdir(path):
            continue
        emb_file = os.path.join(path, "embeddings.npy")
        if not os.path.exists(emb_file):
            continue
        embeddings = np.load(emb_file)
        raw_folder = next(
            (s for s in os.listdir("data/raw")
             if s.lower().replace(" ", "_") == symptom.lower().replace(" ", "_")),
            None
        )
        if raw_folder is None:
            continue
        filenames = [
            f for f in os.listdir(os.path.join("data/raw", raw_folder))
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        gallery[symptom] = {
            "embeddings": embeddings,
            "filenames": filenames,
            "raw_folder": raw_folder
        }
    return gallery

GALLERY = load_gallery()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

ALLOWED_FORMATS = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE_MB = 100

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
    if brightness < 30:
        raise HTTPException(status_code=400, detail="Image is too dark. Please retake in better lighting.")
    if brightness > 240:
        raise HTTPException(status_code=400, detail="Image is too bright. Please avoid direct sunlight.")
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 20:
        raise HTTPException(status_code=400, detail="Image is too blurry. Please retake with a steady hand.")


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/gallery")
def get_gallery():
    symptoms = os.listdir("data/raw")
    return {"symptoms": symptoms}


@router.get("/gallery/{symptom_id}/images")
def get_symptom_images(symptom_id: str):
    raw_dir = "data/raw"
    matched = next(
        (s for s in os.listdir(raw_dir)
         if s.lower().replace(" ", "_") == symptom_id.lower().replace(" ", "_")),
        None
    )
    if matched is None:
        raise HTTPException(status_code=404, detail="Symptom not found")
    images = [
        f for f in os.listdir(os.path.join(raw_dir, matched))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ][:5]
    return {"images": [f"/images/{matched}/{f}" for f in images]}


@router.get("/gallery/{symptom_id}/first-image")
def get_first_image(symptom_id: str):
    raw_dir = "data/raw"
    matched = next(
        (s for s in os.listdir(raw_dir)
         if s.lower().replace(" ", "_") == symptom_id.lower().replace(" ", "_")),
        None
    )
    if matched is None:
        raise HTTPException(status_code=404, detail="Symptom not found")
    images = [
        f for f in os.listdir(os.path.join(raw_dir, matched))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not images:
        raise HTTPException(status_code=404, detail="No images found")
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=f"/images/{matched}/{images[0]}")


@router.get("/gallery/{symptom_id}")
def get_symptom_detail(symptom_id: str):
    raw_dir = "data/raw"
    matched = next(
        (s for s in os.listdir(raw_dir)
         if s.lower().replace(" ", "_") == symptom_id.lower().replace(" ", "_")),
        None
    )
    if matched is None:
        raise HTTPException(status_code=404, detail=f"Symptom '{symptom_id}' not found")
    image_count = len([
        f for f in os.listdir(os.path.join(raw_dir, matched))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    return SymptomDetail(
        name=matched,
        description=f"Reference gallery for {matched} symptom",
        severity="high" if image_count < 20 else "medium",
        action=f"Found {image_count} reference images in gallery"
    )


@router.post("/compare", response_model=CompareResponse)
async def compare_image(file: UploadFile = File(...)):
    contents = await file.read()
    validate_image(contents, file.content_type)

    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        query_emb = MODEL(img_tensor).squeeze().numpy()

    scores = []
    for symptom, data in GALLERY.items():
        for i, ref_emb in enumerate(data["embeddings"]):
            sim = cosine_similarity(query_emb, ref_emb)
            filename = data["filenames"][i] if i < len(data["filenames"]) else None
            scores.append((sim, symptom, filename, data["raw_folder"]))

    scores.sort(reverse=True)

    best_sim = round(float(scores[0][0]), 1)
    second_sim = round(float(scores[1][0]), 1) if len(scores) > 1 else 0
    gap = best_sim - second_sim

    if best_sim < 70 and gap < 5:
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

    best_symptom = scores[0][1]
    best_sim = round(float(scores[0][0]), 1)

    if best_symptom.lower().replace(" ", "_") == "leaf_healthy":
        severity = "low"
        action = "No Action Needed — crop appears healthy"
    elif best_sim >= 80:
        severity = "high"
        action = "Take Action — consult an expert immediately"
    elif best_sim >= 60:
        severity = "medium"
        action = "Monitor — observe and recheck in a few days"
    else:
        severity = "low"
        action = "Low Concern — likely not a match"

    raw_folder = next(
        (s for s in os.listdir("data/raw")
         if s.lower().replace(" ", "_") == best_symptom.lower().replace(" ", "_")),
        best_symptom
    )

    symptom_scores = [
        (sim, fname, folder) for sim, sym, fname, folder in scores
        if sym == best_symptom and fname is not None
    ][:10]

    random.shuffle(symptom_scores)
    top_3 = symptom_scores[:3]

    matches = []
    for i, (sim, fname, folder) in enumerate(top_3):
        reference_image = f"/images/{folder}/{fname}"
        matches.append(MatchResult(
            symptom=best_symptom,
            similarity_pct=best_sim,
            severity=severity,
            description=f"Reference image {i + 1} for {folder.replace('_', ' ')}",
            action=action,
            reference_image=reference_image
        ))

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