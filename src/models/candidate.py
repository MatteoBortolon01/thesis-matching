from pydantic import BaseModel
from src.models.skill import Skill
from typing import List, Optional


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
    raw_text: Optional[str] = None