"""
Job Agent (LLM-based)
Agente autonomo che analizza Job Description e costruisce requisiti normalizzati.

Responsabilità:
- Usa LLMService per estrarre e classificare requisiti
- Normalizza con ESCOMapper
- Produce JobRequirements normalizzati
- Può RILASSARE requisiti durante la negoziazione
"""

from typing import List, Optional

from src.services.llm_service import LLMService
from src.services.esco_mapper import ESCOMapper
from src.models.skill import Skill
from src.models.job import JobRequirements
from src.models.candidate import Language


class JobAgent:
    """
    Agente che analizza Job Description e produce requisiti normalizzati.
    
    Usa LLMService per estrazione + ESCOMapper per normalizzazione.
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        esco_mapper: Optional[ESCOMapper] = None,
        min_mapping_confidence: float = 0.6,
        verbose: bool = False
    ):
        self.min_mapping_confidence = min_mapping_confidence
        self.verbose = verbose
        
        self._llm_service = llm_service
        self._esco_mapper = esco_mapper
    
    @property
    def llm_service(self) -> LLMService:
        if self._llm_service is None:
            self._log("Initializing LLMService...")
            self._llm_service = LLMService()
        return self._llm_service
    
    @property
    def esco_mapper(self) -> ESCOMapper:
        if self._esco_mapper is None:
            self._log("Initializing ESCOMapper...")
            self._esco_mapper = ESCOMapper()
        return self._esco_mapper
    
    def analyze(self, job_description: str) -> JobRequirements:
        """
        Analizza una Job Description e produce requisiti normalizzati.
        
        Pipeline:
        1. LLM estrae requisiti strutturati
        2. Pulizia skill
        3. Normalizzazione ESCO
        4. Costruzione JobRequirements
        """
        self._log("="*60)
        self._log("JOB AGENT: Analyzing Job Description")
        self._log("="*60)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Estrazione con LLM
        # ═══════════════════════════════════════════════════════════════
        self._log("\nStep 1: Extracting requirements with LLM...")
        raw_requirements = self.llm_service.extract_job_requirements(job_description)
        
        job_title = raw_requirements.get("job_title", "")
        llm_required = raw_requirements.get("required_skills", [])
        llm_preferred = raw_requirements.get("preferred_skills", [])
        experience_years = raw_requirements.get("experience_years") or 0
        raw_languages_required = raw_requirements.get("languages_required", [])
        raw_languages_preferred = raw_requirements.get("languages_preferred", [])
        location = raw_requirements.get("location")
        remote_policy = raw_requirements.get("remote_policy")
        
        self._log(f"   Title: {job_title}")
        self._log(f"   Required: {len(llm_required)}, Preferred: {len(llm_preferred)}")
        self._log(f"   Experience: {experience_years} years")
        
        # Parse lingue
        languages_required = self._parse_languages(raw_languages_required)
        languages_preferred = self._parse_languages(raw_languages_preferred)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Pulizia
        # ═══════════════════════════════════════════════════════════════
        self._log("\nStep 2: Cleaning skills...")
        cleaned_required = self._clean_skills(llm_required)
        cleaned_preferred = self._clean_skills(llm_preferred)
        self._log(f"   After cleaning: Required {len(cleaned_required)}, Preferred {len(cleaned_preferred)}")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Normalizzazione ESCO
        # ═══════════════════════════════════════════════════════════════
        self._log(f"\nStep 3: ESCO Normalization...")
        normalized_required = self._normalize_skills(cleaned_required, "required")
        normalized_preferred = self._normalize_skills(cleaned_preferred, "preferred")
        
        self._log(f"   Final: Required {len(normalized_required)}, Preferred {len(normalized_preferred)}")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Costruzione requisiti finali
        # ═══════════════════════════════════════════════════════════════
        return JobRequirements(
            job_title=job_title if job_title else None,
            required_skills=normalized_required,
            preferred_skills=normalized_preferred,
            experience_years=experience_years,
            languages_required=languages_required,
            languages_preferred=languages_preferred,
            location=location,
            remote_policy=remote_policy,
            raw_text=job_description
        )
    
    def refine_requirements(
        self,
        job_requirements: JobRequirements,
        gaps: List[str],
        candidate_skills: List[str]
    ) -> JobRequirements:
        """
        REFINEMENT: Rilassa requisiti se ci sono troppi gap.
        
        Chiamato dall'Orchestrator durante la negoziazione.
        Può spostare alcune skill da required a preferred.
        """
        self._log("\nJOB AGENT: Refining requirements...")
        self._log(f"   Current gaps: {gaps}")
        
        if len(gaps) <= 2:
            self._log("   -> Few gaps, no refinement needed")
            return job_requirements
        
        # Chiedi all'LLM quali required potrebbero essere rilassati
        prompt = f"""Job requires these skills that candidate lacks: {', '.join(gaps)}
Candidate has: {', '.join(candidate_skills[:15])}

Which 1-2 missing skills could reasonably be learned on the job or have partial equivalents in candidate's skills?
Consider: similar technologies, transferable skills, learning curve.

Return JSON: {{"relaxable": ["skill1", "skill2"]}}
If none can be relaxed, return: {{"relaxable": []}}

JSON:"""

        result = self.llm_service.generate_json(prompt, temperature=0.2)
        relaxable = result.get("relaxable", []) if result else []
        
        if not relaxable:
            self._log("   -> LLM suggests no skills can be relaxed")
            return job_requirements
        
        self._log(f"   -> LLM suggests relaxing: {relaxable}")
        
        # Sposta skill rilassabili da required a preferred
        new_required = []
        new_preferred = list(job_requirements.preferred_skills)
        
        relaxable_lower = {s.lower() for s in relaxable}
        
        for skill in job_requirements.required_skills:
            skill_name = skill.esco_name or skill.name
            if skill_name.lower() in relaxable_lower:
                skill.category = "preferred"
                new_preferred.append(skill)
                self._log(f"Moved to preferred: {skill_name}")
            else:
                new_required.append(skill)
        
        return JobRequirements(
            job_title=job_requirements.job_title,
            required_skills=new_required,
            preferred_skills=new_preferred,
            experience_years=job_requirements.experience_years,
            languages_required=job_requirements.languages_required,
            languages_preferred=job_requirements.languages_preferred,
            location=job_requirements.location,
            remote_policy=job_requirements.remote_policy,
            raw_text=job_requirements.raw_text
        )
    
    def _clean_skills(self, skills: List[str]) -> List[str]:
        """Pulisce e valida le skill estratte."""
        cleaned = []
        seen = set()
        
        skip_patterns = ["esperienza", "conoscenza", "capacità", "competenza",
                        "experience", "knowledge", "ability", "competence",
                        "buon", "ottim", "eccellent", "forte", "team",
                        "good", "excellent", "strong"]
        
        for skill in skills:
            if not skill or not isinstance(skill, str):
                continue
            
            skill_clean = skill.strip()
            skill_lower = skill_clean.lower()
            
            if len(skill_clean) < 2:
                continue
            if skill_lower in seen:
                continue
            if any(p in skill_lower for p in skip_patterns) and len(skill_clean) < 20:
                continue
            
            seen.add(skill_lower)
            cleaned.append(skill_clean)
        
        return cleaned
    
    def _normalize_skills(self, skills: List[str], category: str) -> List[Skill]:
        """Normalizza skill con ESCOMapper."""
        if not skills:
            return []
        
        normalized = []
        seen_esco_names = set()
        mappings = self.esco_mapper.map_skills_batch(skills)
        
        for skill_name, mapping in zip(skills, mappings):
            skill_id, normalized_name, mapping_confidence, source = mapping
            
            # Deduplica per esco_name
            esco_name_key = (normalized_name or skill_name).lower()
            if esco_name_key in seen_esco_names:
                self._log(f"   DUP {skill_name} -> {normalized_name} (duplicate, skip)")
                continue
            seen_esco_names.add(esco_name_key)
            
            if skill_id is not None and mapping_confidence >= self.min_mapping_confidence:
                normalized.append(Skill(
                    name=skill_name,
                    esco_id=skill_id,
                    esco_name=normalized_name,
                    confidence=mapping_confidence,
                    category=category
                ))
                self._log(f"   MATCH {skill_name} -> {normalized_name} ({mapping_confidence:.0%})")
            elif mapping_confidence > 0.4:
                normalized.append(Skill(
                    name=skill_name,
                    esco_id=None,
                    esco_name=skill_name,
                    confidence=mapping_confidence,
                    category=category
                ))
                self._log(f"   KEEP {skill_name} (unmapped, {mapping_confidence:.0%})")
        
        return normalized
    
    def _parse_languages(self, raw_languages: list) -> List[Language]:
        """Parsa le lingue estratte dall'LLM."""
        languages = []
        
        for lang in raw_languages:
            if isinstance(lang, dict):
                name = lang.get("name", "")
                level = lang.get("level", "Not specified")
            elif isinstance(lang, str):
                parts = lang.strip().split()
                if len(parts) >= 2 and parts[-1].upper() in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                    name = " ".join(parts[:-1])
                    level = parts[-1].upper()
                else:
                    name = lang
                    level = "Not specified"
            else:
                continue
            
            if name:
                languages.append(Language(name=name, level=level))
        
        return languages
    
    def _log(self, message: str) -> None:
        """Conditional logging."""
        if self.verbose:
            print(f"[JobAgent] {message}")
