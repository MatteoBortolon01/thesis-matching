"""
Candidate Agent (LLM-based)
Agente autonomo che analizza CV e costruisce un profilo candidato normalizzato.

Responsabilità:
- Usa LLMService per estrarre info da CV non strutturati
- Usa ESCOMapper per normalizzare le skill
- Prende DECISIONI su quali skill tenere
- Gestisce conflitti e duplicati
- Produce un CandidateProfile pulito
- Può CERCARE skill implicite durante la negoziazione
"""

from typing import List, Dict, Optional
from pathlib import Path

from PyPDF2 import PdfReader

from src.services.llm_service import LLMService
from src.services.esco_mapper import ESCOMapper
from src.models.skill import Skill
from src.models.candidate import CandidateProfile, Language


class CandidateAgent:
    """
    Agente che analizza CV e produce profili candidato normalizzati.
    
    Usa LLMService per estrazione + ESCOMapper per normalizzazione.
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        esco_mapper: Optional[ESCOMapper] = None,
        min_mapping_confidence: float = 0.6,
        enable_deduplication: bool = True,
        verbose: bool = False,
        allow_inferred_soft_skills: bool = False
    ):
        self.min_mapping_confidence = min_mapping_confidence
        self.enable_deduplication = enable_deduplication
        self.verbose = verbose
        self.allow_inferred_soft_skills = allow_inferred_soft_skills
        
        self._llm_service = llm_service
        self._esco_mapper = esco_mapper
    
    @property
    def llm_service(self) -> LLMService:
        if self._llm_service is None:
            self._log("Inizializzazione LLMService...")
            self._llm_service = LLMService()
        return self._llm_service
    
    @property
    def esco_mapper(self) -> ESCOMapper:
        if self._esco_mapper is None:
            self._log("Inizializzazione ESCOMapper...")
            self._esco_mapper = ESCOMapper()
        return self._esco_mapper
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Estrae testo da un file PDF."""
        reader = PdfReader(pdf_path)
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return "\n".join(text_parts)
    
    def analyze(self, cv_input: str) -> CandidateProfile:
        """
        Analizza un CV e produce un profilo candidato normalizzato.
        
        Args:
            cv_input: Testo del CV o percorso a file PDF
            
        Returns:
            CandidateProfile con skill normalizzate
        """
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Estrazione testo
        # ═══════════════════════════════════════════════════════════════
        if cv_input.endswith('.pdf') and Path(cv_input).exists():
            self._log("Step 1: Estrazione testo da PDF...")
            cv_text = self._extract_text_from_pdf(cv_input)
        else:
            cv_text = cv_input
            self._log(f"Step 1: Input è già testo ({len(cv_text)} caratteri)")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Estrazione info con LLM
        # ═══════════════════════════════════════════════════════════════
        self._log("Step 2: Estrazione informazioni con LLM...")
        cv_info = self.llm_service.extract_cv_info(cv_text)
        
        name = cv_info.get("name")
        experience_years = cv_info.get("experience_years", 0)
        education = cv_info.get("education")
        technical_skills = cv_info.get("technical_skills", [])
        soft_skills = cv_info.get("soft_skills", [])
        certifications = cv_info.get("certifications", [])
        raw_languages = cv_info.get("languages", [])
        
        self._log(f"   -> Nome: {name}, Esperienza: {experience_years} anni")
        self._log(f"   -> Tech skills: {len(technical_skills)}, Soft skills: {len(soft_skills)}")
        
        # Parse lingue
        languages = self._parse_languages(raw_languages)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Costruzione lista skill raw
        # ═══════════════════════════════════════════════════════════════
        self._log("Step 3: Costruzione lista skill...")
        raw_skills = []
        
        for skill_info in technical_skills:
            skill_name = skill_info.get("name", "") if isinstance(skill_info, dict) else str(skill_info)
            if skill_name:
                raw_skills.append(Skill(name=skill_name, category="technical_skill", confidence=1.0))
        
        for skill_info in soft_skills:
            skill_name = skill_info.get("name", "") if isinstance(skill_info, dict) else str(skill_info)
            if skill_name:
                raw_skills.append(Skill(name=skill_name, category="soft_skill", confidence=1.0))
        
        for cert in certifications:
            if cert:
                raw_skills.append(Skill(name=str(cert), category="certification", confidence=1.0))
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Deduplicazione
        # ═══════════════════════════════════════════════════════════════
        if self.enable_deduplication:
            self._log("Step 4: Deduplicazione...")
            deduplicated_skills = self._deduplicate_skills(raw_skills)
            self._log(f"   -> {len(deduplicated_skills)}/{len(raw_skills)} skill uniche")
        else:
            deduplicated_skills = raw_skills
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Normalizzazione ESCO
        # ═══════════════════════════════════════════════════════════════
        self._log("Step 5: Normalizzazione ESCO...")
        normalized_skills = self._normalize_skills(deduplicated_skills)
        self._log(f"   -> {len(normalized_skills)} skill normalizzate")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 6: Costruzione profilo finale
        # ═══════════════════════════════════════════════════════════════
        education_list = []
        if education:
            if isinstance(education, str):
                education_list = [education] if education.strip() else []
            elif isinstance(education, list):
                education_list = [e for e in education if e and isinstance(e, str)]
        
        return CandidateProfile(
            name=name,
            skills=normalized_skills,
            experience_years=experience_years,
            education=education_list,
            languages=languages,
            raw_text=cv_text[:500] + "..." if len(cv_text) > 500 else cv_text
        )
    
    def _deduplicate_skills(self, skills: List[Skill]) -> List[Skill]:
        """Rimuove duplicati (case-insensitive)."""
        seen = set()
        deduplicated = []
        
        for skill in skills:
            key = skill.name.lower().strip()
            if key not in seen:
                seen.add(key)
                deduplicated.append(skill)
        
        return deduplicated
    
    def _normalize_skills(self, skills: List[Skill]) -> List[Skill]:
        """Normalizza skill con ESCOMapper."""
        normalized = []
        skill_names = [s.name for s in skills]
        mappings = self.esco_mapper.map_skills_batch(skill_names)
        
        for skill, mapping in zip(skills, mappings):
            skill_id, normalized_name, mapping_confidence, source = mapping
            
            if skill_id is not None and mapping_confidence >= self.min_mapping_confidence:
                normalized.append(Skill(
                    name=skill.name,
                    esco_id=skill_id,
                    esco_name=normalized_name,
                    confidence=mapping_confidence,
                    category=skill.category,
                    level=skill.level
                ))
                self._log(f"   MATCH {skill.name} -> {normalized_name} ({mapping_confidence:.0%})")
            elif mapping_confidence > 0.4:
                # Tieni skill con confidence decente anche se non mappata
                normalized.append(Skill(
                    name=skill.name,
                    esco_id=None,
                    esco_name=skill.name,
                    confidence=mapping_confidence,
                    category=skill.category,
                    level=skill.level
                ))
                self._log(f"   KEEP {skill.name} (non mappata, {mapping_confidence:.0%})")
        
        return normalized
    
    def _parse_languages(self, raw_languages: list) -> List[Language]:
        """Parsa le lingue estratte dall'LLM."""
        languages = []
        
        for lang in raw_languages:
            if isinstance(lang, dict):
                name = lang.get("name", "")
                level = lang.get("level", "Non specificato")
            elif isinstance(lang, str):
                parts = lang.strip().split()
                if len(parts) >= 2 and parts[-1].upper() in ["A1", "A2", "B1", "B2", "C1", "C2"]:
                    name = " ".join(parts[:-1])
                    level = parts[-1].upper()
                else:
                    name = lang
                    level = "Non specificato"
            else:
                continue
            
            if name:
                languages.append(Language(name=name, level=level))
        
        return languages
    
    def refine_profile(
        self,
        candidate_profile: CandidateProfile,
        gaps: List[str],
        job_required_skills: List[str]
    ) -> CandidateProfile:
        """
        REFINEMENT: Cerca skill implicite nel CV che potrebbero coprire i gap.
        
        Chiamato dall'Orchestrator durante la negoziazione.
        Ri-analizza il CV cercando skill correlate ai gap.
        """
        self._log("\nCANDIDATE AGENT: Refining profile...")
        self._log(f"   Current gaps: {gaps}")
        
        if len(gaps) == 0:
            self._log("   -> No gaps to address")
            return candidate_profile
        
        # Chiedi all'LLM di cercare skill implicite nel CV
        cv_text = candidate_profile.raw_text or ""
        current_skills = [s.esco_name or s.name for s in candidate_profile.skills]
        
        prompt = f"""CV excerpt: {cv_text[:1500]}

Current extracted skills: {', '.join(current_skills[:20])}

Missing skills needed for job: {', '.join(gaps)}

Look for IMPLICIT or RELATED skills in the CV that weren't explicitly extracted but could partially cover the gaps.
Examples:
- If "CI/CD" is missing but CV mentions "automated deployments" → that's implicit CI/CD
- If "Docker" is missing but CV mentions "containerization" → that's implicit Docker
- If "SQL" is missing but CV mentions "database queries" → that's implicit SQL

Return JSON with skills found: {{"implicit_skills": ["skill1", "skill2"]}}
If no implicit skills found, return: {{"implicit_skills": []}}

JSON:"""

        result = self.llm_service.generate_json(prompt, temperature=0.2)
        implicit_skills = result.get("implicit_skills", []) if result else []
        
        if not implicit_skills:
            self._log("   -> No implicit skills found")
            return candidate_profile
        
        self._log(f"   -> Found implicit skills: {implicit_skills}")
        
        # Normalizza e aggiungi le skill implicite
        new_skills = list(candidate_profile.skills)
        current_names = {s.name.lower() for s in new_skills}
        
        for skill_name in implicit_skills:
            if skill_name.lower() in current_names:
                continue
            
            # Mappa con ESCO
            mappings = self.esco_mapper.map_skills_batch([skill_name])
            if mappings:
                skill_id, normalized_name, confidence, source = mappings[0]
                if confidence >= 0.5:
                    new_skill = Skill(
                        name=skill_name,
                        esco_id=skill_id,
                        esco_name=normalized_name,
                        confidence=confidence * 0.8,  # Confidence ridotta perché implicita
                        category="implicit"
                    )
                    new_skills.append(new_skill)
                    self._log(f"Added implicit: {skill_name} -> {normalized_name}")
        
        return CandidateProfile(
            name=candidate_profile.name,
            skills=new_skills,
            experience_years=candidate_profile.experience_years,
            education=candidate_profile.education,
            languages=candidate_profile.languages,
            raw_text=candidate_profile.raw_text
        )
    
    def _log(self, message: str) -> None:
        if self.verbose:
            print(f"[CandidateAgent] {message}")
