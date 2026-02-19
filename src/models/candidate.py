from pydantic import BaseModel
from src.models.skill import Skill
from typing import List, Optional, Dict, Any


class Language(BaseModel):
    """Lingua parlata dal candidato."""
    name: str                           # Nome lingua (es. "Italiano")
    level: Optional[str] = None         # Livello (es. "Madrelingua", "B2", "Fluente")


class CandidateProfile(BaseModel):
    name: Optional[str] = None
    skills: List[Skill] = []
    experience_years: int = 0
    education: Optional[List[str]] = None
    languages: List[Language] = []      # Lingue parlate
    job_title: Optional[str] = None
    certifications: List[str] = []
    llm_extracted: Optional[Dict[str, Any]] = None  # JSON grezzo restituito da extract_cv_info
    raw_text: Optional[str] = None
