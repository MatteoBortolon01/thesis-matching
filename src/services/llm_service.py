"""
LLM Service
Wrapper per LLM locali (Ollama o LM Studio) - gestisce le chiamate al modello.
"""

import json
import os
import re
from typing import Optional, List, Dict, Any

import ollama

from src.services.logging_utils import log_section, print_with_prefix

class OllamaNotAvailableError(Exception):
    """Eccezione per quando Ollama non è disponibile."""
    pass


class LLMService:
    """
    Servizio per interagire con LLM locali (Ollama o LM Studio).
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.3,
        timeout: int = 120,
        num_gpu: int = -1,  # -1 = auto (usa tutte le GPU disponibili)
        provider: str = "lmstudio",  # "ollama" o "lmstudio"
        lmstudio_base_url: Optional[str] = None,
        lmstudio_api_key: Optional[str] = None,
        lmstudio_model: Optional[str] = "meta-llama-3.1-8b-instruct"
    ):
        """
        Inizializza il servizio LLM.
        
        Args:
            model: Nome del modello (Ollama o LM Studio)
            temperature: Temperatura per la generazione (0-1, più basso = più deterministico)
            timeout: Timeout in secondi per le chiamate
            num_gpu: Numero di layer GPU (-1 = auto, 0 = solo CPU)
            provider: "ollama" o "lmstudio"
            lmstudio_base_url: Base URL per LM Studio (default: env LMSTUDIO_BASE_URL o http://localhost:1234/v1)
            lmstudio_api_key: API key per LM Studio (default: env LMSTUDIO_API_KEY o "lmstudio")
            lmstudio_model: Nome modello LM Studio (fallback su "model")
        """
        self.provider = (provider or "ollama").lower()
        if self.provider not in {"ollama", "lmstudio"}:
            raise ValueError("provider deve essere 'ollama' o 'lmstudio'")

        self.model = lmstudio_model or model
        self.temperature = temperature
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.is_available = False
        self._lmstudio_client = None
        self.lmstudio_base_url = lmstudio_base_url or os.getenv("LMSTUDIO_BASE_URL", "http://bears.disi.unitn.it:1234/v1")
        self.lmstudio_api_key = lmstudio_api_key or os.getenv("LMSTUDIO_API_KEY", "lmstudio")
        
        # Verifica provider
        if self.provider == "ollama":
            self._check_ollama()
        else:
            self._check_lmstudio()
    
    def _check_ollama(self) -> None:
        """Verifica che Ollama sia in esecuzione e il modello sia disponibile."""
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models] if models.models else []
            
            # Cerca il modello (con o senza tag :latest)
            model_found = any(
                self.model in name or name.startswith(self.model)
                for name in model_names
            )
            
            if not model_found:
                self._log(f"Modello '{self.model}' non trovato. Modelli disponibili: {model_names}")
                self._log(f"Esegui: ollama pull {self.model}")
                self.is_available = False
            else:
                self.is_available = True
                self._log(f"LLM Service pronto (modello: {self.model})")
                
        except ConnectionError:
            self.is_available = False
            self._log("Ollama non raggiungibile. Avvialo con: ollama serve")
        except Exception as e:
            self.is_available = False
            self._log(f"Errore connessione Ollama: {e}")
            self._log("Assicurati che Ollama sia in esecuzione (ollama serve)")

    def _get_lmstudio_client(self):
        """Crea (lazy) client OpenAI compatibile con LM Studio."""
        if self._lmstudio_client is None:
            try:
                from openai import OpenAI
            except Exception as e:
                raise OllamaNotAvailableError(
                    "Package 'openai' mancante. Installa con: pip install openai"
                ) from e

            self._lmstudio_client = OpenAI(
                base_url=self.lmstudio_base_url,
                api_key=self.lmstudio_api_key
            )
        return self._lmstudio_client

    def _check_lmstudio(self) -> None:
        """Verifica che LM Studio sia raggiungibile e il modello disponibile."""
        try:
            client = self._get_lmstudio_client()
            models = client.models.list()
            model_names = [m.id for m in models.data] if getattr(models, "data", None) else []

            model_found = any(
                self.model in name or name.startswith(self.model)
                for name in model_names
            )

            if not model_found:
                self._log(f"Modello '{self.model}' non trovato su LM Studio. Modelli disponibili: {model_names}")
                self.is_available = False
            else:
                self.is_available = True
                self._log(f"LLM Service pronto (provider: lmstudio, modello: {self.model})")
        except Exception as e:
            self.is_available = False
            self._log(f"LM Studio non raggiungibile: {e}")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Genera testo dal prompt.
        
        Args:
            prompt: Il prompt da inviare al modello
            system_prompt: Prompt di sistema opzionale
            temperature: Override della temperatura
            
        Returns:
            Testo generato dal modello
        """
        if not self.is_available:
            if self.provider == "ollama":
                raise OllamaNotAvailableError(
                    "Ollama non disponibile. Avvialo con 'ollama serve' e assicurati "
                    f"che il modello '{self.model}' sia installato (ollama pull {self.model})"
                )
            raise OllamaNotAvailableError(
                "LM Studio non disponibile. Avvialo e assicurati che l'endpoint sia raggiungibile "
                f"(base_url={self.lmstudio_base_url}) e che il modello '{self.model}' sia caricato"
            )
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            if self.provider == "ollama":
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature or self.temperature,
                        "num_gpu": self.num_gpu  # -1 = usa tutte le GPU
                    }
                )
                return response.message.content

            client = self._get_lmstudio_client()
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if self.provider == "ollama":
                raise OllamaNotAvailableError(f"Errore chiamata Ollama: {e}")
            raise OllamaNotAvailableError(f"Errore chiamata LM Studio: {e}")
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Genera e parsa JSON dal prompt.
        
        Args:
            prompt: Il prompt che richiede output JSON
            system_prompt: Prompt di sistema opzionale
            temperature: Override della temperatura
            
        Returns:
            Dizionario parsato o None se parsing fallisce
        """
        # Aggiungi istruzione per JSON se non presente
        if "json" not in prompt.lower():
            prompt += "\n\nRispondi SOLO con JSON valido, senza altro testo."
        
        response_text = self.generate(prompt, system_prompt, temperature)
        
        # Prova a estrarre JSON dalla risposta
        return self._extract_json(response_text)
    
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Estrae JSON da testo che potrebbe contenere altro."""
        # Prima prova parsing diretto
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass
        
        # Cerca il primo JSON object {...} bilanciato
        # Usando conteggio parentesi per trovare la chiusura corretta
        start_idx = text.find('{')
        if start_idx == -1:
            return None
        
        depth = 0
        end_idx = start_idx
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        
        if depth == 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                pass
        
        # Fallback: cerca in blocchi di codice
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group(1))
                except (json.JSONDecodeError, IndexError):
                    continue
        
        return None
    
    def extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """
        Estrae i requisiti da una job description.
        
        Args:
            job_description: Testo della job description
            
        Returns:
            Dizionario con required_skills, preferred_skills, experience_years
        """
        system_prompt = """You are an expert HR analyst specialized in parsing job descriptions.
Extract information precisely, conservatively, and in a structured format.
Only use information explicitly stated in the text; do not infer or embellish.
ALWAYS respond with valid JSON only and follow the requested schema exactly."""

        prompt = f"""Analyze this job description and extract the requirements.

JOB DESCRIPTION:
{job_description}

Respond with this JSON format:
{{
    "job_title": "position title",
    "required_skills": ["skill1", "skill2", ...],
    "preferred_skills": ["skill3", "skill4", ...],
    "experience_years": minimum_years_required,
    "languages_required": [{{"name": "English", "level": "B2"}}],
    "languages_preferred": [{{"name": "German", "level": "A2"}}],
    "location": "city or office location",
    "remote_policy": "full_remote" | "hybrid" | "on_site",
    "notes": "any important notes"
}}

RULES:
- "required_skills": MANDATORY skills (keywords: "required", "must have", "necessary", "essential")
- "preferred_skills": OPTIONAL skills (keywords: "preferred", "nice to have", "plus", "bonus")
- Include BOTH technical skills and soft skills if explicitly mentioned
- Soft skills examples: communication, teamwork, leadership, problem solving, time management
- Do NOT include years of experience or seniority phrases as skills (e.g., "3+ years of backend development experience")
- If a requirement mixes years and a skill, extract ONLY the skill (e.g., "3+ years of Python" -> "Python")
- If not specified, consider skills as required
- "experience_years": integer, 0 if not specified
- Extract SPECIFIC skills, avoid vague phrases (e.g., "strong background")
- "languages_required": MANDATORY languages with level
- "languages_preferred": optional/preferred languages
- "remote_policy": "full_remote", "hybrid", "on_site" or null if not specified
- If a field is not mentioned, use an empty value ("" or [] or 0 or null)

JSON:"""

        result = self.generate_json(prompt, system_prompt, temperature=0.1)

        # Log the full JSON response in a dedicated section
        log_section(self._log, "LLM RAW JOB REQUIREMENTS JSON", width=60, char="=")
        self._log(json.dumps(result, ensure_ascii=False, indent=2) if result else "<No JSON parsed>")
        self._log("=" * 60)

        # Valori di default se parsing fallisce
        if result is None:
            return {
                "job_title": "",
                "required_skills": [],
                "preferred_skills": [],
                "experience_years": 0,
                "languages_required": [],
                "languages_preferred": [],
                "location": None,
                "remote_policy": None,
                "notes": "Parsing fallito"
            }

        return result
    
    def extract_cv_info(self, cv_text: str) -> Dict[str, Any]:
        """
        Estrae tutte le informazioni rilevanti da un CV.
        
        Args:
            cv_text: Testo del CV
            
        Returns:
            Dizionario con name, experience_years, education, technical_skills, soft_skills
        """
        system_prompt = """You are a CV Parsing Agent used in an automated candidate screening system.

Your task is to extract factual information explicitly stated in the CV,
without making assumptions or inferences.

You must be precise, conservative, and consistent.

Constraints:
- Extract ONLY information explicitly mentioned in the CV
- Do NOT infer missing details
- Do NOT reinterpret or normalize job titles, degrees, or skills
- Do NOT add explanations, comments, or extra text
- Always output valid JSON strictly following the requested schema

If a field cannot be determined from the CV, use an appropriate empty value
(e.g., empty string, empty list, or 0)."""

        prompt = f"""TASK:
Analyze the following CV/resume and extract structured information.

CV:
{cv_text}

OUTPUT FORMAT (JSON ONLY):
{{
    "name": "candidate full name or empty string",
    "experience_years": number,
    "education": "most recent or relevant degree, or empty string",
    "job_title": "current or most recent role, or empty string",
    "technical_skills": [
        {{ "name": "skill name", "category": "category" }}
    ],
    "soft_skills": [
        {{ "name": "soft skill name", "category": "soft_skill" }}
    ],
    "certifications": ["certification name"],
    "languages": [
        {{ "name": "language", "level": "level" }}
    ]
}}

SKILLS EXTRACTION RULES:
- Extract ONLY skills explicitly mentioned in the CV
- Include both technical and non-technical professional skills
- Extract individual skills (e.g. "Excel, PowerPoint" → two entries)
- Search ALL sections (Skills, Tools, Competencies, Experience, etc.)
- Avoid generic or vague descriptions (e.g. "professional experience")

TECHNICAL SKILLS CATEGORIES:
Use the most appropriate category from the following:
- programming_language
- framework
- database
- cloud
- devops
- tool
- software
- methodology
- other

Examples:
- Excel → tool
- SAP → software
- AutoCAD → software
- Python → programming_language
- Agile → methodology

SOFT SKILLS RULES:
- Include interpersonal, organizational, and cognitive skills
- Examples: communication, teamwork, leadership, time management, problem solving
- Extract ONLY if explicitly mentioned

LANGUAGES RULES:
- Extract ONLY languages explicitly mentioned
- Use the proficiency level stated in the CV
- If the level is missing, use "Not specified"
- Do NOT infer native language
- If no languages are mentioned, return []

EXPERIENCE RULES:
- Estimate total professional experience in years
- Include non-technical roles
- If the candidate has no professional experience, return 0

CERTIFICATIONS RULE:
- If a certification is explicitly mentioned,
  extract it ONLY as a certification
- Do NOT extract its internal modules as separate skills
- Do NOT infer skills from certifications

IMPORTANT:
- Output JSON only
- No comments
- No markdown
- No additional text

JSON:"""

        result = self.generate_json(prompt, system_prompt, temperature=0.1)

        # Log the full JSON response in a dedicated section
        log_section(self._log, "LLM RAW CANDIDATE PROFILE JSON", width=60, char="=")
        self._log(json.dumps(result, ensure_ascii=False, indent=2) if result else "<No JSON parsed>")
        self._log("=" * 60)

        # Valori di default se parsing fallisce
        if result is None:
            return {
                "name": None,
                "experience_years": 0,
                "education": None,
                "job_title": None,
                "technical_skills": [],
                "soft_skills": [],
                "certifications": [],
                "languages": []
            }

        return result
    
    def generate_match_explanation(
        self,
        score: float,
        matched_skills: List[str],
        missing_required: List[str],
        missing_preferred: List[str],
        candidate_experience: int,
        required_experience: int,
        strengths: Optional[List[str]] = None
    ) -> str:
        """
        Genera una spiegazione del match.
        
        Args:
            score: Score del match (0-100)
            matched_skills: Skill che matchano
            missing_required: Skill required mancanti
            missing_preferred: Skill preferred mancanti
            candidate_experience: Anni esperienza candidato
            required_experience: Anni esperienza richiesti
            strengths: Punti di forza aggiuntivi
            
        Returns:
            Spiegazione in linguaggio naturale
        """
        # Prompt compatto in inglese
        matched_str = ', '.join(matched_skills[:10]) if matched_skills else 'none'
        gaps_str = ', '.join(missing_required[:5]) if missing_required else 'none'
        
        prompt = f"""Match score: {score:.0f}/100. 
Matched skills: {matched_str}. 
Missing required: {gaps_str}. 
Experience: candidate {candidate_experience}y / required {required_experience}y.

Write a brief explanation (2-3 sentences in Italian) covering:
1. Overall compatibility assessment
2. Main strengths
3. Key gaps (if any)

Rules:
- If candidate_experience >= required_experience, treat it as a strength, not a gap
- If candidate_experience < required_experience, mention it as a potential gap
- Do not present higher experience as a negative
- Keep tone concise and professional."""

        return self.generate(prompt, temperature=0.3)
    
    def reason_skill_equivalence(
        self,
        gap_skills: List[str],
        candidate_skills: List[str],
        max_retries: int = 2,
        verbose: bool = False,
        system_prompt: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Ragiona sulle equivalenze tra skill richieste e skill candidato.
        Con retry automatico se risposta vuota o malformata.
        """
        if not gap_skills or not candidate_skills:
            return {}
        
        gaps_str = ', '.join(gap_skills[:10])
        cand_str = ', '.join(candidate_skills[:30])
        
        system_prompt = """You are a Skill Equivalence Evaluator.

Your task is to determine whether two technical skills can be considered
functionally equivalent in a professional recruiting context.

You must be extremely conservative:
- False positives are unacceptable
- If an equivalence is not clearly justified, you must reject it
- When in doubt, choose NOT equivalent"""

        # Prompt in inglese (LLM performa meglio)
        prompt = f"""TASK:
Evaluate whether any REQUIRED skills can be considered technically equivalent
to skills already possessed by the candidate.

CONTEXT:
- Required skills (gaps): [{gaps_str}]
- Candidate skills: [{cand_str}]

EQUIVALENCE CRITERIA (ALL must be satisfied):
Two skills may be considered equivalent ONLY IF:
1. They belong to the SAME technical domain
2. They serve the SAME primary functional role
3. Knowledge is directly transferable with minimal retraining
4. They are commonly interchangeable in real-world job requirements

NON-EQUIVALENCE RULES:
The following are NEVER equivalent:
- Backend frameworks ↔ Frontend frameworks
- Programming languages ↔ Frameworks
- Databases ↔ Caches
- Infrastructure tools ↔ Application frameworks
- CI/CD tools ↔ Runtime or serverless platforms

DECISION PROCESS (internal):
For each potential match:
- Identify domain
- Identify functional role
- Assess transferability
- Decide equivalence or rejection

OUTPUT FORMAT (JSON ONLY):
Return a flat JSON mapping:
"required_skill" → "equivalent_candidate_skill"

Example:
{{"FastAPI": "Flask"}}

If no valid equivalences exist, return:
{{}}

IMPORTANT:
- Do NOT force equivalences
- Do NOT generalize by buzzwords
- Do NOT infer seniority or experience
- Reject partial or conceptual similarity

JSON:"""

        candidate_lower = {s.lower(): s for s in candidate_skills}

        def extract_equivalences(result: dict, candidate_lower: dict) -> Dict[str, str]:
            """
            Extract validated skill equivalences from a strict LLM JSON response.
            Expected format:
            {
              "matches": {
                "GapSkill": "CandidateSkill"
              }
            }
            """
            if not isinstance(result, dict):
                return {}

            matches = result.get("matches")
            if not isinstance(matches, dict):
                return {}

            valid: Dict[str, str] = {}

            for gap, match in matches.items():
                if not isinstance(gap, str) or not isinstance(match, str):
                    continue

                match_l = match.lower()

                # Exact match only (conservative)
                if match_l in candidate_lower:
                    valid[gap] = candidate_lower[match_l]

            return valid

        for attempt in range(max_retries):
            response = self.generate_json(prompt, system_prompt=system_prompt, temperature=0.1)

            if not response:
                continue

            matches = extract_equivalences(response, candidate_lower)
            if matches:
                return matches

        return {}

    def _log(self, message: str) -> None:
        print_with_prefix("[LLMService]", message, enabled=True)
    

