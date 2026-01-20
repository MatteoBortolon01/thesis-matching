from pydantic import BaseModel
from src.models.skill import Skill
from src.models.candidate import Language
from typing import List, Optional


class JobRequirements(BaseModel):
    """Requisiti estratti da una Job Description."""
    job_title: Optional[str] = None
    required_skills: List[Skill] = []
    preferred_skills: List[Skill] = []
    experience_years: int = 0
    languages_required: List[Language] = []   # Lingue richieste (es. Inglese B2)
    languages_preferred: List[Language] = []  # Lingue gradite
    location: Optional[str] = None            # Sede di lavoro
    remote_policy: Optional[str] = None       # "full_remote", "hybrid", "on_site"
    raw_text: Optional[str] = None