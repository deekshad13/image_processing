from  pydantic import BaseModel
from typing import List

class MatchResult(BaseModel):
    symptom: str
    severity: str 
    description: str 
    action: str
    reference_image: str
    similarity_pct: float

class CompareResponse(BaseModel):
    status: str
    matches: List[MatchResult]
    plant_part_detected: str

class SymptomDetail(BaseModel):
    name:str
    description: str
    severity: str
    action: str