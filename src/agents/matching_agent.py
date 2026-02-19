"""
Matching Agent
Agente autonomo che calcola il match tra candidato e job requirements.

Responsabilità:
- Confronta skill per ESCO ID (match esatto)
- Fallback su similarità semantica per skill non mappate
- Calcola score aggregato con pesi configurabili
- Genera spiegazione del match con LLM
- DECIDE come pesare i vari fattori
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from src.services.llm_service import LLMService, OllamaNotAvailableError
from src.services.esco_mapper import ESCOMapper
from src.services.logging_utils import log_section, print_with_prefix
from src.models.skill import Skill
from src.models.candidate import CandidateProfile, Language
from src.models.job import JobRequirements
from src.models.match_result import MatchResult


class MatchingAgent:
    """
    Agente che calcola il match tra un candidato e i requisiti di un job.
    
    LOGICA DI MATCHING:
    1. Match skill per ESCO ID (preciso)
    2. Match skill per nome (fuzzy)
    3. Verifica esperienza
    4. Calcola score aggregato
    5. Genera spiegazione con LLM
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        esco_mapper: Optional[ESCOMapper] = None,
        weight_required: float = 0.7,
        weight_preferred: float = 0.2,
        weight_experience: float = 0.1,
        weight_languages: float = 0.0,
        weight_title: float = 0.0,
        fuzzy_match_threshold: float = 0.75,
        verbose: bool = False
    ):
        self.weight_required = weight_required
        self.weight_preferred = weight_preferred
        self.weight_experience = weight_experience
        self.weight_languages = weight_languages
        self.weight_title = weight_title
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.verbose = verbose
        
        self._llm_service = llm_service
        self._esco_mapper = esco_mapper
    
    @property
    def llm_service(self) -> LLMService:
        if self._llm_service is None:
            self._llm_service = LLMService()
        return self._llm_service
    
    @property
    def esco_mapper(self) -> ESCOMapper:
        if self._esco_mapper is None:
            self._esco_mapper = ESCOMapper()
        return self._esco_mapper
    
    def match(self, candidate: CandidateProfile, job: JobRequirements) -> MatchResult:
        """Calcola il match tra candidato e job."""
        self._log(f"Matching: {candidate.name or 'Candidato'} vs {job.job_title or 'Job'}")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Match skill REQUIRED
        # ═══════════════════════════════════════════════════════════════
        log_section(self._log, "Step 1: Match skill REQUIRED", width=60, char="-")
        required_matches, required_gaps, required_match_types = self._match_skills(
            candidate.skills, job.required_skills
        )
        
        n_required = len(job.required_skills)
        n_required_matched = len(required_matches)
        required_score = (n_required_matched / n_required * 100) if n_required > 0 else 100
        self._log(f"   -> Matched: {n_required_matched}/{n_required} ({required_score:.0f}%)")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Match skill PREFERRED
        # ═══════════════════════════════════════════════════════════════
        log_section(self._log, "Step 2: Match skill PREFERRED", width=60, char="-")
        preferred_matches, preferred_gaps, preferred_match_types = self._match_skills(
            candidate.skills, job.preferred_skills
        )
        
        n_preferred = len(job.preferred_skills)
        n_preferred_matched = len(preferred_matches)
        preferred_score = (n_preferred_matched / n_preferred * 100) if n_preferred > 0 else 100
        self._log(f"   -> Matched: {n_preferred_matched}/{n_preferred} ({preferred_score:.0f}%)")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Check esperienza
        # ═══════════════════════════════════════════════════════════════
        log_section(self._log, "Step 3: Check esperienza", width=60, char="-")
        experience_score = self._calculate_experience_score(
            candidate.experience_years, job.experience_years
        )
        self._log(f"   -> {candidate.experience_years}/{job.experience_years} anni -> {experience_score:.0f}%")

        # Segnali opzionali (non cambiano lo score se i pesi restano a 0)
        languages_score = 100.0
        title_score = 100.0
        if self.weight_languages > 0:
            languages_score = self._calculate_languages_score(
                candidate.languages, job.languages_required, job.languages_preferred
            )
        if self.weight_title > 0:
            title_score = self._calculate_title_score(candidate.job_title, job.job_title)
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Calcola score aggregato
        # ═══════════════════════════════════════════════════════════════
        total_weight = (
            self.weight_required +
            self.weight_preferred +
            self.weight_experience +
            self.weight_languages +
            self.weight_title
        ) or 1.0

        w_req = self.weight_required / total_weight
        w_pref = self.weight_preferred / total_weight
        w_exp = self.weight_experience / total_weight
        w_lang = self.weight_languages / total_weight
        w_title = self.weight_title / total_weight
        
        final_score = (
            required_score * w_req +
            preferred_score * w_pref +
            experience_score * w_exp +
            languages_score * w_lang +
            title_score * w_title
        )
        self._log(f"SCORE FINALE: {final_score:.1f}/100")
        
        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Genera spiegazione
        # ═══════════════════════════════════════════════════════════════
        strengths = self._identify_strengths(
            candidate, job, required_matches, preferred_matches, experience_score
        )
        
        # Include preferred gaps in final output (marked as preferred)
        preferred_gaps_marked = [f"(preferred) {g}" for g in preferred_gaps]
        final_gaps_out = required_gaps + preferred_gaps_marked

        explanation = self._generate_explanation(
            score=final_score,
            matched_skills=required_matches + preferred_matches,
            missing_required=required_gaps,
            missing_preferred=preferred_gaps,
            candidate_experience=candidate.experience_years,
            required_experience=job.experience_years,
            strengths=strengths
        )

        # Combina match_types da required e preferred
        all_match_types = {**required_match_types, **preferred_match_types}

        return MatchResult(
            score=round(final_score, 1),
            matched_skills=required_matches + preferred_matches,
            gaps=final_gaps_out,
            strengths=strengths,
            explanation=explanation,
            match_types=all_match_types,
            required_score=round(required_score, 1),
            preferred_score=round(preferred_score, 1),
            experience_score=round(experience_score, 1),
        )
    
    def _match_skills(
        self,
        candidate_skills: List[Skill],
        job_skills: List[Skill]
    ) -> Tuple[List[str], List[str]]:
        """
        Match skill candidato vs job.
        
        Strategia:
        1. Match per ESCO ID
        2. Match per nome normalizzato
        3. Match per nome originale
        4. Similarità embedding (fallback)
        5. LLM reasoning per equivalenze semantiche
        """
        matched = []
        match_types: Dict[str, str] = {}  # skill_display -> tipo di match
        pending_gaps = []  # Gap temporanei, verranno processati con LLM
        pending_gap_skills = []  # Skill objects per LLM
        
        candidate_ids = {s.esco_id for s in candidate_skills if s.esco_id}
        candidate_names_lower = {s.name.lower() for s in candidate_skills}
        candidate_esco_names_lower = {s.esco_name.lower() for s in candidate_skills if s.esco_name}
        
        for job_skill in job_skills:
            skill_display = job_skill.esco_name or job_skill.name
            match_found = False
            match_type = ""
            
            # Match 1: Per ESCO ID
            if job_skill.esco_id and job_skill.esco_id in candidate_ids:
                match_found = True
                match_type = "esco_id"
                self._log(f"   MATCH {skill_display} (ESCO ID)")
            
            # Match 2: Per nome normalizzato
            elif job_skill.esco_name and job_skill.esco_name.lower() in candidate_esco_names_lower:
                match_found = True
                match_type = "esco_name"
                self._log(f"   MATCH {skill_display} (nome ESCO)")
            
            # Match 3: Per nome originale
            elif job_skill.name.lower() in candidate_names_lower:
                match_found = True
                match_type = "original_name"
                self._log(f"   MATCH {skill_display} (nome originale)")
            
            # Match 4: Per nome in esco_names del candidato
            elif job_skill.name.lower() in candidate_esco_names_lower:
                match_found = True
                match_type = "reverse"
                self._log(f"   MATCH {skill_display} (reverse)")
            
            # Match 5: Similarità embedding
            if not match_found:
                match_found, similarity = self._fuzzy_match(job_skill.name, candidate_skills)
                if match_found:
                    match_type = "fuzzy"
                    self._log(f"   MATCH {skill_display} (fuzzy {similarity:.0%})")
            
            if match_found:
                matched.append(skill_display)
                match_types[skill_display] = match_type
            else:
                pending_gaps.append(skill_display)
                pending_gap_skills.append(job_skill)
        
        # ═══════════════════════════════════════════════════════════════
        # LLM REASONING: Per i gap, chiediamo all'LLM equivalenze semantiche
        # ═══════════════════════════════════════════════════════════════
        final_gaps = []
        if pending_gaps:
            self._log(f"LLM reasoning per {len(pending_gaps)} gap...")
            
            # Crea lista gap con sia nome originale che ESCO per miglior matching
            gap_names_for_llm = []
            gap_name_to_display = {}  # Mappa per ritrovare il display name
            for skill in pending_gap_skills:
                # Usa nome originale (più corto, es "AWS" invece di "Amazon Web Services")
                gap_names_for_llm.append(skill.name)
                gap_name_to_display[skill.name.lower()] = skill.esco_name or skill.name
                # Aggiungi anche esco_name se diverso
                if skill.esco_name and skill.esco_name != skill.name:
                    gap_names_for_llm.append(skill.esco_name)
                    gap_name_to_display[skill.esco_name.lower()] = skill.esco_name
            
            llm_matches = self._llm_skill_reasoning(gap_names_for_llm, candidate_skills)
            
            # Processa risultati
            matched_displays = set()
            for gap_skill in pending_gaps:
                gap_lower = gap_skill.lower()
                # Cerca match per questo gap o per il suo nome alternativo
                found_match = None
                for llm_gap, llm_match in llm_matches.items():
                    if llm_gap.lower() == gap_lower or gap_name_to_display.get(llm_gap.lower()) == gap_skill:
                        found_match = llm_match
                        break
                
                if found_match and gap_skill not in matched_displays:
                    matched.append(gap_skill)
                    match_types[gap_skill] = "llm_reasoning"
                    matched_displays.add(gap_skill)
                    self._log(f"   MATCH {gap_skill} <- LLM({found_match})")
                else:
                    final_gaps.append(gap_skill)
                    self._log(f"   GAP {gap_skill}")
        
        # Deduplica mantenendo ordine
        matched = list(dict.fromkeys(matched))
        final_gaps = list(dict.fromkeys(final_gaps))
        
        return matched, final_gaps, match_types
    
    def _llm_skill_reasoning(
        self,
        gap_skills: List[str],
        candidate_skills: List[Skill]
    ) -> Dict[str, str]:
        """
        Usa LLM per trovare equivalenze semantiche tra skill.
        Es: Flask ≈ FastAPI, GCP ≈ AWS, MySQL ≈ PostgreSQL
        """
        if not gap_skills:
            return {}
        
        # Includi sia nome originale che nome ESCO per dare più contesto all'LLM
        candidate_names = set()
        for s in candidate_skills:
            # Salta soft skills
            if s.category and s.category.lower() in {"soft_skill", "soft"}:
                continue
            # Salta skill che sembrano soft skills dal nome
            if any(soft in s.name.lower() for soft in ["teamwork", "problem solving", "communication", "leadership"]):
                continue
            candidate_names.add(s.name)
            if s.esco_name and s.esco_name != s.name:
                candidate_names.add(s.esco_name)
        candidate_names = list(candidate_names)
        
        if self.verbose:
            self._log(
                "   [DEBUG] Skill equivalence reasoning input: "
                f"{len(gap_skills)} gaps, {len(candidate_names)} candidate skills (soft skills excluded)"
            )
        
        try:
            raw_matches = self.llm_service.reason_skill_equivalence(
                gap_skills, candidate_names, verbose=self.verbose
            )
        except Exception as e:
            if self.verbose:
                self._log(f"   [DEBUG] LLM error: {e}")
            return {}
        
        # Valida risultati (anti-allucinazione)
        candidate_lower = {s.lower() for s in candidate_names}
        
        return {
            k: v for k, v in raw_matches.items()
            if v and isinstance(v, str) and v.lower() in candidate_lower
        }
    
    def _fuzzy_match(self, job_skill_name: str, candidate_skills: List[Skill]) -> Tuple[bool, float]:
        """Match fuzzy usando similarità semantica."""
        job_embedding = self.esco_mapper.model.encode(job_skill_name, convert_to_numpy=True)
        best_similarity = 0.0
        
        for candidate_skill in candidate_skills:
            candidate_name = candidate_skill.esco_name or candidate_skill.name
            candidate_embedding = self.esco_mapper.model.encode(candidate_name, convert_to_numpy=True)
            
            similarity = np.dot(job_embedding, candidate_embedding) / (
                np.linalg.norm(job_embedding) * np.linalg.norm(candidate_embedding)
            )
            best_similarity = max(best_similarity, similarity)
        
        return best_similarity >= self.fuzzy_match_threshold, best_similarity
    
    def _calculate_experience_score(self, candidate_years: int, required_years: int) -> float:
        """Calcola score esperienza."""
        if required_years == 0:
            return 100.0
        
        if candidate_years >= required_years:
            return 100.0
        else:
            return (candidate_years / required_years) * 100

    def _calculate_languages_score(
        self,
        candidate_languages: List[Language],
        required_languages: List[Language],
        preferred_languages: List[Language]
    ) -> float:
        """Calcola uno score 0-100 basato sul matching dei nomi lingua (livelli ignorati)."""
        if not required_languages and not preferred_languages:
            return 100.0

        candidate_names = {
            self._normalize_lang_name(l.name)
            for l in (candidate_languages or [])
            if getattr(l, "name", None)
        }

        def _count_matches(target: List[Language]) -> tuple[int, int]:
            if not target:
                return 0, 0
            target_names = [
                self._normalize_lang_name(l.name)
                for l in target
                if getattr(l, "name", None)
            ]
            matched = sum(1 for n in target_names if n in candidate_names)
            return matched, len(target_names)

        req_matched, req_total = _count_matches(required_languages or [])
        pref_matched, pref_total = _count_matches(preferred_languages or [])

        required_score = (req_matched / req_total * 100.0) if req_total > 0 else 100.0
        preferred_score = (pref_matched / pref_total * 100.0) if pref_total > 0 else 100.0

        if req_total > 0 and pref_total > 0:
            return required_score * 0.8 + preferred_score * 0.2
        if req_total > 0:
            return required_score
        return preferred_score

    def _calculate_title_score(self, candidate_title: Optional[str], job_title: Optional[str]) -> float:
        """Calcola uno score 0-100 usando similarità embedding (neutral=100 se titolo mancante)."""
        if not candidate_title or not job_title:
            return 100.0

        try:
            cand_emb = self.esco_mapper.model.encode(candidate_title, convert_to_numpy=True)
            job_emb = self.esco_mapper.model.encode(job_title, convert_to_numpy=True)
            similarity = float(np.dot(cand_emb, job_emb) / (np.linalg.norm(cand_emb) * np.linalg.norm(job_emb)))
            similarity = max(0.0, similarity)
            return min(100.0, similarity * 100.0)
        except Exception:
            return 100.0

    def _normalize_lang_name(self, name: str) -> str:
        return (name or "").strip().lower()
    
    def _identify_strengths(
        self,
        candidate: CandidateProfile,
        job: JobRequirements,
        required_matches: List[str],
        preferred_matches: List[str],
        experience_score: float
    ) -> List[str]:
        """Identifica i punti di forza del candidato."""
        strengths = []
        
        if len(required_matches) == len(job.required_skills) and len(job.required_skills) > 0:
            strengths.append("Tutte le skill richieste presenti")
        
        if len(preferred_matches) > len(job.preferred_skills) * 0.5:
            strengths.append(f"{len(preferred_matches)} skill preferenziali matchate")
        
        if experience_score >= 100:
            strengths.append(f"Esperienza adeguata ({candidate.experience_years} anni)")
        
        if candidate.languages:
            lang_names = [l.name for l in candidate.languages[:2]]
            strengths.append(f"Lingue: {', '.join(lang_names)}")
        
        return strengths
    
    def _generate_explanation(
        self,
        score: float,
        matched_skills: List[str],
        missing_required: List[str],
        missing_preferred: List[str],
        candidate_experience: int,
        required_experience: int,
        strengths: List[str]
    ) -> str:
        """Genera spiegazione del match."""
        try:
            return self.llm_service.generate_match_explanation(
                score=score,
                matched_skills=matched_skills,
                missing_required=missing_required,
                missing_preferred=missing_preferred,
                candidate_experience=candidate_experience,
                required_experience=required_experience
            )
        except OllamaNotAvailableError:
            # Fallback locale
            parts = []
            
            if score >= 80:
                parts.append(f"Ottima compatibilità ({score:.0f}/100).")
            elif score >= 60:
                parts.append(f"Buona compatibilità ({score:.0f}/100).")
            else:
                parts.append(f"Compatibilità parziale ({score:.0f}/100).")
            
            if matched_skills:
                parts.append(f"Skill in comune: {', '.join(matched_skills[:5])}.")
            
            if missing_required:
                parts.append(f"Skill mancanti: {', '.join(missing_required[:3])}.")
            
            if strengths:
                parts.append(f"Punti di forza: {strengths[0]}.")
            
            return " ".join(parts)
    
    def _log(self, message: str) -> None:
        print_with_prefix("[MatchingAgent]", message, enabled=self.verbose)
