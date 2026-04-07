from pydantic import BaseModel
from typing import List, Dict


class VerifyResult(BaseModel):
    symptom_id: str
    symptom_name: str
    similarity_pct: float
    confidence: str
    recommendation: str
    gallery_size: int
    top_n_used: int


class VerifyResponse(BaseModel):
    status: str
    result: VerifyResult
    thresholds: Dict[str, str]


class GallerySymptom(BaseModel):
    symptom_id: str
    display_name: str
    image_count: int


class GalleryResponse(BaseModel):
    symptoms: List[GallerySymptom]
    total_images: int


class SymptomDetail(BaseModel):
    symptom_id: str
    display_name: str
    description: str
    image_count: int


class ValidationSettings(BaseModel):
    max_file_size_mb: float
    min_brightness: float
    max_brightness: float
    min_laplacian_var: float


class SimilaritySettings(BaseModel):
    top_n_mean: int
    threshold_high: float
    threshold_medium: float


class SettingsResponse(BaseModel):
    validation: ValidationSettings
    similarity: SimilaritySettings
