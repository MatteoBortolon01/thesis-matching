from pydantic import BaseModel
from typing import Optional

class Skill(BaseModel):
    name: str
    esco_id: Optional[str] = None
    esco_name: Optional[str] = None
    confidence: float = 1.0
    category: Optional[str] = None
    level: Optional[str] = None