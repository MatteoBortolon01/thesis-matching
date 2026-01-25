"""
Matching Orchestrator
Coordina i tre agenti e gestisce il ciclo di negoziazione.

Responsabilità:
- Inizializza e coordina JobAgent, CandidateAgent, MatchingAgent
- Gestisce il REFINEMENT LOOP quando lo score è basso
- Produce il risultato finale con log della negoziazione
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from src.agents.job_agent import JobAgent
from src.agents.candidate_agent import CandidateAgent
from src.agents.matching_agent import MatchingAgent
from src.services.llm_service import LLMService
from src.services.esco_mapper import ESCOMapper
from src.services.logging_utils import log_section, print_with_prefix
from src.models.job import JobRequirements
from src.models.candidate import CandidateProfile
from src.models.match_result import MatchResult


@dataclass
class NegotiationRound:
    """Rappresenta un round di negoziazione."""
    round_number: int
    action: str
    agent: str
    details: str
    score_before: float
    score_after: float


@dataclass 
class OrchestratorResult:
    """Risultato completo dell'orchestrazione."""
    match_result: MatchResult
    job_requirements: JobRequirements
    candidate_profile: CandidateProfile
    negotiation_log: List[NegotiationRound] = field(default_factory=list)
    total_rounds: int = 1
    final_score: float = 0.0
    initial_score: float = 0.0
    score_improvement: float = 0.0


class MatchingOrchestrator:
    """
    Orchestratore che coordina i tre agenti e gestisce la negoziazione.
    
    FLUSSO:
    1. JobAgent analizza la JD → JobRequirements
    2. CandidateAgent analizza il CV → CandidateProfile
    3. MatchingAgent calcola il match → MatchResult
    4. SE score < threshold → REFINEMENT LOOP
       a. CandidateAgent cerca skill implicite
       b. JobAgent rilassa alcuni requisiti
       c. MatchingAgent ricalcola
    5. Ritorna risultato finale con log negoziazione
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        esco_mapper: Optional[ESCOMapper] = None,
        refinement_threshold: float = 50.0,
        max_refinement_rounds: int = 1,
        verbose: bool = False
    ):
        self.refinement_threshold = refinement_threshold
        self.max_refinement_rounds = max_refinement_rounds
        self.verbose = verbose
        
        # Servizi condivisi
        self._llm_service = llm_service
        self._esco_mapper = esco_mapper
        
        # Agenti (lazy init)
        self._job_agent = None
        self._candidate_agent = None
        self._matching_agent = None
    
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
    
    @property
    def job_agent(self) -> JobAgent:
        if self._job_agent is None:
            self._job_agent = JobAgent(
                llm_service=self.llm_service,
                esco_mapper=self.esco_mapper,
                verbose=self.verbose
            )
        return self._job_agent
    
    @property
    def candidate_agent(self) -> CandidateAgent:
        if self._candidate_agent is None:
            self._candidate_agent = CandidateAgent(
                llm_service=self.llm_service,
                esco_mapper=self.esco_mapper,
                verbose=self.verbose
            )
        return self._candidate_agent
    
    @property
    def matching_agent(self) -> MatchingAgent:
        if self._matching_agent is None:
            self._matching_agent = MatchingAgent(
                llm_service=self.llm_service,
                esco_mapper=self.esco_mapper,
                verbose=self.verbose
            )
        return self._matching_agent
    
    def run(
        self,
        cv_input: str,
        job_description: str,
        enable_refinement: bool = True
    ) -> OrchestratorResult:
        """
        Esegue il matching completo con eventuale negoziazione.
        
        Args:
            cv_input: Testo del CV o percorso PDF
            job_description: Testo della Job Description
            enable_refinement: Se True, attiva il refinement loop
            
        Returns:
            OrchestratorResult con match e log negoziazione
        """
        negotiation_log = []
        
        log_section(self._log, "MATCHING ORCHESTRATOR: Starting Analysis", width=70, char="=")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 1: Initial Analysis
        # ═══════════════════════════════════════════════════════════════
        log_section(self._log, "PHASE 1: Initial Analysis", width=70, char="-")
        
        # Job Agent analizza JD
        self._log("Calling JobAgent...")
        job_requirements = self.job_agent.analyze(job_description)
        self._log(f"   -> Job: {job_requirements.job_title}")
        self._log(f"   -> Required: {len(job_requirements.required_skills)} skills")
        self._log(f"   -> Preferred: {len(job_requirements.preferred_skills)} skills")
        
        # Candidate Agent analizza CV
        self._log("Calling CandidateAgent...")
        candidate_profile = self.candidate_agent.analyze(cv_input)
        self._log(f"   -> Candidate: {candidate_profile.name}")
        self._log(f"   -> Skills: {len(candidate_profile.skills)}")
        self._log(f"   -> Experience: {candidate_profile.experience_years} years")
        # Log candidate languages if available
        try:
            langs = candidate_profile.languages or []
            if langs:
                langs_str = ", ".join(
                    f"{lang.name} ({lang.level})" if getattr(lang, 'level', None) else f"{lang.name}"
                    for lang in langs
                )
            else:
                langs_str = "-"
        except Exception:
            langs_str = "-"
        self._log(f"   -> Languages: {langs_str}")
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: Initial Matching
        # ═══════════════════════════════════════════════════════════════
        log_section(self._log, "PHASE 2: Initial Matching", width=70, char="-")
        
        self._log("Calling MatchingAgent...")
        match_result = self.matching_agent.match(candidate_profile, job_requirements)
        initial_score = match_result.score
        
        self._log(f"   -> Initial Score: {initial_score:.1f}/100")
        self._log(f"   -> Gaps: {len(match_result.gaps)}")
        
        negotiation_log.append(NegotiationRound(
            round_number=0,
            action="initial_match",
            agent="MatchingAgent",
            details=f"Initial analysis complete",
            score_before=0,
            score_after=initial_score
        ))
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 3: Refinement Loop (if enabled and score is low)
        # ═══════════════════════════════════════════════════════════════
        current_score = initial_score
        rounds_done = 0
        
        if enable_refinement and current_score < self.refinement_threshold and match_result.gaps:
            log_section(
                self._log,
                f"PHASE 3: Negotiation (score {current_score:.1f} < {self.refinement_threshold})",
                width=70,
                char="-",
            )
            
            for round_num in range(1, self.max_refinement_rounds + 1):
                self._log(f"ROUND {round_num}")
                score_before = current_score
                gaps = match_result.gaps
                
                # ─── Step A: CandidateAgent cerca skill implicite ───
                self._log("CandidateAgent: Looking for implicit skills...")
                candidate_skills_names = [s.esco_name or s.name for s in candidate_profile.skills]
                job_required_names = [s.esco_name or s.name for s in job_requirements.required_skills]
                
                refined_candidate = self.candidate_agent.refine_profile(
                    candidate_profile,
                    gaps,
                    job_required_names
                )
                
                skills_added = len(refined_candidate.skills) - len(candidate_profile.skills)
                if skills_added > 0:
                    self._log(f"   Added {skills_added} implicit skills")
                    candidate_profile = refined_candidate
                    
                    negotiation_log.append(NegotiationRound(
                        round_number=round_num,
                        action="find_implicit_skills",
                        agent="CandidateAgent",
                        details=f"Found {skills_added} implicit skills",
                        score_before=score_before,
                        score_after=score_before  # Will update after re-match
                    ))
                
                # ─── Step B: JobAgent rilassa requisiti ───
                self._log("JobAgent: Considering requirement relaxation...")
                candidate_skills_names = [s.esco_name or s.name for s in candidate_profile.skills]
                
                refined_job = self.job_agent.refine_requirements(
                    job_requirements,
                    gaps,
                    candidate_skills_names
                )
                
                skills_relaxed = len(job_requirements.required_skills) - len(refined_job.required_skills)
                if skills_relaxed > 0:
                    self._log(f"   Relaxed {skills_relaxed} requirements")
                    job_requirements = refined_job
                    
                    negotiation_log.append(NegotiationRound(
                        round_number=round_num,
                        action="relax_requirements",
                        agent="JobAgent",
                        details=f"Moved {skills_relaxed} skills to preferred",
                        score_before=score_before,
                        score_after=score_before
                    ))
                
                # ─── Step C: Ricalcola match ───
                self._log("MatchingAgent: Recalculating...")
                match_result = self.matching_agent.match(candidate_profile, job_requirements)
                current_score = match_result.score
                
                self._log(f"   -> New Score: {current_score:.1f}/100 (was {score_before:.1f})")
                
                # Update last negotiation log with new score
                if negotiation_log:
                    negotiation_log[-1].score_after = current_score
                
                negotiation_log.append(NegotiationRound(
                    round_number=round_num,
                    action="recalculate",
                    agent="MatchingAgent",
                    details=f"Score: {score_before:.1f} -> {current_score:.1f}",
                    score_before=score_before,
                    score_after=current_score
                ))
                
                rounds_done = round_num
                
                # Se score è migliorato abbastanza, ferma
                if current_score >= self.refinement_threshold:
                    self._log("Threshold reached, stopping refinement")
                    break
                
                # Se non c'è miglioramento, ferma
                if current_score <= score_before:
                    self._log("No improvement, stopping refinement")
                    break
        
        # ═══════════════════════════════════════════════════════════════
        # PHASE 4: Final Result
        # ═══════════════════════════════════════════════════════════════
        log_section(self._log, "FINAL RESULT", width=70, char="-")
        
        score_improvement = current_score - initial_score
        
        self._log(f"   Initial Score: {initial_score:.1f}")
        self._log(f"   Final Score:   {current_score:.1f}")
        self._log(f"   Improvement:   {score_improvement:+.1f}")
        self._log(f"   Rounds:        {rounds_done + 1}")
        
        return OrchestratorResult(
            match_result=match_result,
            job_requirements=job_requirements,
            candidate_profile=candidate_profile,
            negotiation_log=negotiation_log,
            total_rounds=rounds_done + 1,
            final_score=current_score,
            initial_score=initial_score,
            score_improvement=score_improvement
        )
    
    def _log(self, message: str) -> None:
        """Conditional logging."""
        print_with_prefix("[Orchestrator]", message, enabled=self.verbose)


# ═══════════════════════════════════════════════════════════════════════════
# SIMPLE API
# ═══════════════════════════════════════════════════════════════════════════

def match_cv_to_job(
    cv_input: str,
    job_description: str,
    verbose: bool = False,
    enable_refinement: bool = True
) -> OrchestratorResult:
    """
    API semplice per matching CV-Job.
    
    Args:
        cv_input: Testo CV o path PDF
        job_description: Testo JD
        verbose: Se True, stampa log
        enable_refinement: Se True, attiva negoziazione
        
    Returns:
        OrchestratorResult completo
    """
    orchestrator = MatchingOrchestrator(verbose=verbose)
    return orchestrator.run(cv_input, job_description, enable_refinement)
