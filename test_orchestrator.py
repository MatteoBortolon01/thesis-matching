"""
Test Orchestrator 
"""

from src.orchestrator import match_cv_to_job, MatchingOrchestrator

# ═══════════════════════════════════════════════════════════════════════════
# TEST DATA
# ═══════════════════════════════════════════════════════════════════════════

TEST_CV = """
Name: Marco Rossi
Location: Milano, Italy
Role: Backend Engineer
Years of experience: 5

Professional summary:
Backend engineer with 5 years of experience building APIs and microservices in Python.
Strong focus on reliability, testing, and collaboration.

Experience:
Backend Engineer, PayLine SRL (2022 - present)
- Built REST APIs with FastAPI and PostgreSQL
- Implemented background jobs and integrations
- Wrote tests with pytest and maintained GitHub Actions pipelines
- Deployed services using Docker and Kubernetes on AWS (ECS, RDS, S3)

Software Engineer, CloudBridge (2020 - 2022)
- Developed Python services with Django and FastAPI
- Maintained CI/CD pipelines and Linux deployments
- Worked with Redis and RabbitMQ

Technical skills:
- Python, FastAPI, Django
- PostgreSQL, Redis
- Docker, Kubernetes
- AWS (ECS, RDS, S3)
- Git, GitHub Actions
- Linux
- REST APIs, pytest
- Terraform (basic)

Soft skills:
- teamwork
- communication
- problem solving
- time management
- mentoring

Education:
MSc in Computer Engineering, Politecnico di Milano

Languages:
- Italian: Native
- English: B2
"""

TEST_JD = """
BACKEND DEVELOPER - Fintech Startup

Stiamo cercando un Backend Developer esperto per il nostro team.

REQUISITI OBBLIGATORI:
- Python (minimo 3 anni)
- FastAPI o Django
- PostgreSQL
- Docker e Kubernetes
- CI/CD (Jenkins o GitHub Actions)
- Esperienza con microservizi

REQUISITI PREFERENZIALI:
- AWS o GCP
- Redis
- GraphQL

ESPERIENZA: minimo 3 anni

SEDE: Milano, ibrido
"""
TEST_JD1 = """Title: Backend Engineer (Python)
Company: FinTechWave S.p.A.
Location: Milano
Remote policy: Hybrid (2 giorni in ufficio)

Summary:
We are looking for a Backend Engineer to build and maintain scalable APIs for our payment platform.

Responsibilities:
- Design and implement REST APIs
- Maintain microservices and background jobs
- Collaborate with product and data teams
- Write tests and support CI/CD pipelines

Requirements (must have):
- 3+ years of backend development experience
- Python
- FastAPI
- PostgreSQL
- REST API design
- Docker
- Git
- Linux
- Testing with pytest
- CI/CD pipelines
- Soft skills: teamwork, communication, problem solving

Preferred (nice to have):
- AWS (ECS, RDS, S3)
- Kubernetes
- Redis
- RabbitMQ
- Terraform
- GraphQL
- Soft skills: leadership, mentoring

Languages:
- English B2 or higher
- Italian B2 or higher

Notes:
We value proactive attitude and clear communication."""

def test_orchestrator():
    """Test completo dell'orchestrator."""
    print("="*70)
    print("TEST ORCHESTRATOR CON NEGOZIAZIONE")
    print("="*70)
    
    # Esegui matching con negoziazione
    result = match_cv_to_job(
        cv_input=TEST_CV,
        job_description=TEST_JD1,
        verbose=True,
        enable_refinement=True
    )
    
    # ═══════════════════════════════════════════════════════════════
    # RISULTATI
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "="*70)
    print("RISULTATI FINALI")
    print("="*70)
    
    print(f"\nCandidato: {result.candidate_profile.name}")
    print(f"Skill: {len(result.candidate_profile.skills)}")
    
    print(f"\nJob: {result.job_requirements.job_title}")
    print(f"Required: {len(result.job_requirements.required_skills)}")
    print(f"Preferred: {len(result.job_requirements.preferred_skills)}")
    
    print(f"\nSCORE")
    print(f"Iniziale: {result.initial_score:.1f}/100")
    print(f"Finale:   {result.final_score:.1f}/100")
    print(f"Δ:        {result.score_improvement:+.1f}")
    
    print(f"\nMatch Details")
    print(f"Matched: {result.match_result.matched_skills[:5]}")
    print(f"Gaps:    {result.match_result.gaps[:5]}")
    
    # ═══════════════════════════════════════════════════════════════
    # LOG NEGOZIAZIONE
    # ═══════════════════════════════════════════════════════════════
    print(f"\nNEGOTIATION LOG ({len(result.negotiation_log)} steps)")
    for entry in result.negotiation_log:
        print(f"Round {entry.round_number}: [{entry.agent}] {entry.action}")
        print(f"   {entry.details}")
        print(f"   Score: {entry.score_before:.1f} → {entry.score_after:.1f}")
    
    # ═══════════════════════════════════════════════════════════════
    # SPIEGAZIONE LLM
    # ═══════════════════════════════════════════════════════════════
    print(f"\nSPIEGAZIONE:")
    print(f" {result.match_result.explanation}")
    
    return result


def test_without_refinement():
    """Test senza negoziazione per confronto."""
    print("\n" + "="*70)
    print("TEST SENZA NEGOZIAZIONE (baseline)")
    print("="*70)
    
    result = match_cv_to_job(
        cv_input=TEST_CV,
        job_description=TEST_JD,
        verbose=False,
        enable_refinement=False
    )
    
    print(f"\n Score: {result.final_score:.1f}/100")
    print(f" Gaps:  {len(result.match_result.gaps)}")
    
    return result


if __name__ == "__main__":
    # Test con negoziazione
    result_with = test_orchestrator()
    
    # Test senza negoziazione  
    # result_without = test_without_refinement()
    
    # Confronto
    print("\n" + "="*70)
    # print("CONFRONTO")
    print("="*70)
    # print(f"   Senza negoziazione: {result_without.final_score:.1f}/100")
    print(f"   Con negoziazione:   {result_with.final_score:.1f}/100")
    # print(f"   Miglioramento:      {result_with.score_improvement:+.1f}")
