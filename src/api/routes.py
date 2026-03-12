from fastapi import APIRouter, UploadFile, File
from src.api.schemas import CompareResponse, MatchResult
from src.model.encoder import load_dinov2, get_embedding
from src.model.similarity import cosine_similarity
import os, io, json, numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
from src.api.schemas import CompareResponse, MatchResult, SymptomDetail
import cv2
import numpy as np
from fastapi import HTTPException

router = APIRouter()

# Load model and gallery once at startup
MODEL = load_dinov2()
EMBEDDINGS_DIR = "data/embeddings"

def load_gallery():
    gallery = {}
    for symptom in os.listdir(EMBEDDINGS_DIR):
        path = os.path.join(EMBEDDINGS_DIR, symptom)
        if not os.path.isdir(path):
            continue
        emb_file = os.path.join(path, "embeddings.npy")
        if os.path.exists(emb_file):
            gallery[symptom] = np.load(emb_file)
    return gallery

GALLERY = load_gallery()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@router.get("/health")
def health_check():
    return {"status": "ok"}

@router.get("/gallery")
def get_gallery():
    symptoms = os.listdir("data/raw")
    return {"symptoms": symptoms}

ALLOWED_FORMATS = {"image/jpeg", "image/png", "image/jpg"}
MAX_FILE_SIZE_MB = 100

def validate_image(contents: bytes, content_type: str):
    # Check file format
    if content_type not in ALLOWED_FORMATS:
        raise HTTPException(status_code=400, detail="Invalid file format. Only JPG and PNG allowed.")

    # Check file size
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB.")

    # Convert to numpy array for image checks
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Could not read image. File may be corrupted.")

    # Check brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    if brightness < 30:
        raise HTTPException(status_code=400, detail="Image is too dark. Please retake in better lighting.")
    if brightness > 240:
        raise HTTPException(status_code=400, detail="Image is too bright. Please avoid direct sunlight.")

    # Check blurriness using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 50:
        raise HTTPException(status_code=400, detail="Image is too blurry. Please retake.")
    
@router.post("/compare", response_model=CompareResponse)
async def compare_image(file: UploadFile = File(...)):
    # Read and convert uploaded image to embedding
    contents = await file.read()
    validate_image(contents, file.content_type)
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        query_emb = MODEL(img_tensor).squeeze().numpy()

    # Compare against gallery
    scores = []
    for symptom, embeddings in GALLERY.items():
        for ref_emb in embeddings:
            sim = cosine_similarity(query_emb, ref_emb)
            scores.append((sim, symptom))

    scores.sort(reverse=True)
    top_matches = scores[:5]

    # Format results
    seen = set()
    matches = []
    for sim, symptom in top_matches:
        if symptom in seen:
            continue
        seen.add(symptom)
        sim_pct = round(float(sim), 1)
        if sim_pct >= 80:
            severity = "high"
            action = "Take Action — consult an expert immediately"
        elif sim_pct >= 60:
            severity = "medium"
            action = "Monitor — observe and recheck in a few days"
        else:
            severity = "low"
            action = "Low Concern — likely not a match"

        matches.append(MatchResult(
            symptom=symptom,
            similarity_pct=sim_pct,
            severity=severity,
            description=f"Matched against {symptom} reference images",
            action=action,
            reference_image=f"/gallery/{symptom}/ref_001.jpg"
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

@router.get("/gallery/{symptom_id}")
def get_symptom_detail(symptom_id: str):
    raw_dir = "data/raw"
    matched = next(
        (s for s in os.listdir(raw_dir)
         if s.lower().replace(" ", "_") == symptom_id.lower().replace(" ", "_")),
        None
    )
    if matched is None:
        from fastapi import HTTPException
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