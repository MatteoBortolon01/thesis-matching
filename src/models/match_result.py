from pydantic import BaseModel
from typing import List

class MatchResult(BaseModel):
    score: float  # 0-100
    matched_skills: List[str]
    gaps: List[str]
    strengths: List[str]
    explanation: str