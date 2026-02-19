from pydantic import BaseModel
from typing import Dict, List

class MatchResult(BaseModel):
    score: float  # 0-100
    matched_skills: List[str]
    gaps: List[str]
    strengths: List[str]
    explanation: str
    # Breakdown per tipo di match: skill_name -> tipo
    # Tipi: "esco_id", "esco_name", "original_name", "reverse", "fuzzy", "llm_reasoning"
    match_types: Dict[str, str] = {}
    # Breakdown score per componente
    required_score: float = 0.0
    preferred_score: float = 0.0
    experience_score: float = 0.0