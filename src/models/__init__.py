# models package
"""Data models for the job matching system."""

from src.models.skill import Skill
from src.models.candidate import CandidateProfile, Language
from src.models.job import JobRequirements
from src.models.match_result import MatchResult

__all__ = [
    "Skill",
    "CandidateProfile",
    "Language",
    "JobRequirements",
    "MatchResult",
]
