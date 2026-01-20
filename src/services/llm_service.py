"""
LLM Service
Wrapper per Ollama - gestisce le chiamate al modello LLM locale.
"""

import ollama
import json
import re
from typing import Optional, List, Dict, Any


class OllamaNotAvailableError(Exception):
    """Eccezione per quando Ollama non è disponibile."""
    pass


class LLMService:
    """
    Servizio per interagire con Ollama (LLM locale).
    """
    
    def __init__(
        self,
        model: str = "llama3.2",
        temperature: float = 0.3,
        timeout: int = 120,
        num_gpu: int = -1  # -1 = auto (usa tutte le GPU disponibili)
    ):
        """
        Inizializza il servizio LLM.
        
        Args:
            model: Nome del modello Ollama (default: llama3.2)
            temperature: Temperatura per la generazione (0-1, più basso = più deterministico)
            timeout: Timeout in secondi per le chiamate
            num_gpu: Numero di layer GPU (-1 = auto, 0 = solo CPU)
        """
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.num_gpu = num_gpu
        self.is_available = False
        
        # Verifica che Ollama sia disponibile
        self._check_ollama()
    
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
            raise OllamaNotAvailableError(
                "Ollama non disponibile. Avvialo con 'ollama serve' e assicurati "
                f"che il modello '{self.model}' sia installato (ollama pull {self.model})"
            )
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature or self.temperature,
                    "num_gpu": self.num_gpu  # -1 = usa tutte le GPU
                }
            )
            return response.message.content
        except Exception as e:
            raise OllamaNotAvailableError(f"Errore chiamata Ollama: {e}")
    
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
Extract information precisely and in a structured format.
ALWAYS respond with valid JSON only."""

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
- If not specified, consider skills as required
- "experience_years": integer, 0 if not specified
- Extract SPECIFIC technical skills (languages, frameworks, tools), not generic descriptions
- "languages_required": MANDATORY languages with level
- "languages_preferred": optional/preferred languages
- "remote_policy": "full_remote", "hybrid", "on_site" or null if not specified

JSON:"""

        result = self.generate_json(prompt, system_prompt, temperature=0.1)
        
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
        system_prompt = """You are an expert recruiter who analyzes CVs/resumes.
Extract information precisely and in a structured format.
ALWAYS respond with valid JSON only, no comments or other text."""

        prompt = f"""Analyze this CV/resume and extract all relevant information.

CV:
{cv_text}

Respond with this JSON format:
{{
    "name": "candidate full name",
    "experience_years": total_years_of_experience,
    "education": "most recent/relevant degree",
    "job_title": "current or most recent role",
    "technical_skills": [
        {{"name": "skill name", "category": "category"}},
        ...
    ],
    "soft_skills": [
        {{"name": "soft skill name", "category": "soft_skill"}},
        ...
    ],
    "certifications": ["cert1", "cert2", ...],
    "languages": [
        {{"name": "Italian", "level": "Native"}},
        {{"name": "English", "level": "B2"}}
    ]
}}

RULES for skills:
- "technical_skills": programming languages, frameworks, databases, tools, technologies, cloud platforms
- Search in ALL sections: Skills, Tools, Languages, Technologies, Frameworks, Cloud, DevOps, etc.
- Cloud platforms (AWS, GCP, Azure) go in "cloud" category
- "soft_skills": interpersonal, organizational, problem solving, teamwork, communication skills
- "category" for technical: "programming_language", "framework", "database", "cloud", "devops", "tool", "other"
- Extract SPECIFIC skills, not generic descriptions
- If candidate mentions "React, Node.js" extract two separate skills
- If there's a "Tools" section, extract EVERY tool listed (GCP, Docker, Git, etc.)
- "experience_years": estimate total years of work experience (0 if student with no experience)

RULES for languages:
- "languages": list of objects with "name" and "level"
- "level": use the level indicated in the CV (e.g., "Native", "Fluent", "B1", "B2", "C1", "C2", "Basic")
- If level not specified, use "Not specified"
- IMPORTANT: extract ONLY languages EXPLICITLY mentioned in the CV
- DO NOT invent or assume languages (e.g., don't assume Italian just because CV is in Italian)
- If no languages mentioned, return empty list []

JSON:"""

        result = self.generate_json(prompt, system_prompt, temperature=0.1)
        
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

Be concise and professional."""

        return self.generate(prompt, temperature=0.3)
    
    def reason_skill_equivalence(
        self,
        gap_skills: List[str],
        candidate_skills: List[str],
        max_retries: int = 2,
        verbose: bool = False
    ) -> Dict[str, str]:
        """
        Ragiona sulle equivalenze tra skill richieste e skill candidato.
        Con retry automatico se risposta vuota o malformata.
        """
        if not gap_skills or not candidate_skills:
            return {}
        
        gaps_str = ', '.join(gap_skills[:10])
        cand_str = ', '.join(candidate_skills[:30])
        
        # Prompt in inglese (LLM performa meglio)
        prompt = f"""Required skills (gaps): [{gaps_str}]
Candidate has: [{cand_str}]

VALID EQUIVALENCES (same technology category ONLY):
- Python web frameworks (backend): FastAPI = Flask = Django
- SQL databases: PostgreSQL = MySQL = SQL Server = SQL
- Cloud providers: AWS = GCP = Azure
- Container orchestration: Kubernetes = Docker Swarm
- Message queues: RabbitMQ = Redis = Kafka

INVALID - NEVER match these (different categories):
- Backend ≠ Frontend: Django ≠ React, Flask ≠ Vue, FastAPI ≠ Angular
- Serverless ≠ CI/CD: AWS Lambda ≠ Jenkins, Lambda ≠ GitHub Actions
- Database ≠ Cache: PostgreSQL ≠ Redis
- Container ≠ Orchestration: Docker ≠ Kubernetes

RULES:
1. Match ONLY if technologies serve the SAME purpose
2. Django/Flask/FastAPI are BACKEND - never match with React/Vue/Angular
3. If unsure, do NOT include the match
4. Return ONLY valid technical equivalences

Return flat JSON. Example: {{"FastAPI":"Flask","PostgreSQL":"SQL","AWS":"GCP"}}
If no valid equivalence, return: {{}}

JSON:"""

        candidate_lower = {s.lower(): s for s in candidate_skills}

        def _flatten_equivalences(obj: Any) -> Dict[str, str]:
            """Flatten possible shapes returned by the LLM into gap->match string.

            Handles:
            - {"Gap": {"FastAPI": "Flask"}}
            - {"FastAPI": "Flask"}
            - {"Python web frameworks": {"FastAPI": "Flask", "Django": "Flask"}}
            - strings like "FastAPI = Flask" or lists of such strings
            - nested single-key wrappers
            """
            out: Dict[str, str] = {}

            if isinstance(obj, dict):
                for k, v in obj.items():
                    # If value is a simple string mapping: {"FastAPI": "Flask"}
                    if isinstance(v, str):
                        out[k] = v
                        continue

                    # If value is a dict, inner keys are likely specific gaps
                    if isinstance(v, dict):
                        for inner_k, inner_v in v.items():
                            if isinstance(inner_v, str):
                                out[inner_k] = inner_v
                            elif isinstance(inner_v, list):
                                # take first string-like element
                                for el in inner_v:
                                    if isinstance(el, str):
                                        out[inner_k] = el
                                        break
                        continue

                    # If value is a list of strings like ["FastAPI = Flask"]
                    if isinstance(v, list):
                        for el in v:
                            if isinstance(el, str) and "=" in el:
                                left, right = map(str.strip, el.split("=", 1))
                                out[left] = right
                        continue

            # If obj itself is a string like "FastAPI = Flask"
            if isinstance(obj, str):
                if "=" in obj:
                    left, right = map(str.strip, obj.split("=", 1))
                    out[left] = right

            return out

        for attempt in range(max_retries + 1):
            try:
                temp = 0.1 + (attempt * 0.1)
                response = self.generate(prompt, temperature=temp)

                if verbose:
                    self._log(f"   [LLM attempt {attempt+1}] Response: {response[:400]}...")

                result = self._extract_json(response)

                if result is None:
                    if verbose:
                        self._log(f"   [LLM attempt {attempt+1}] Failed to extract JSON")
                    continue

                # Try several known wrapper keys
                candidate_map: Dict[str, str] = {}
                if isinstance(result, dict):
                    # common wrappers
                    for wrapper in ("Gap", "gaps", "equivalences", "equivalence", "matches"):
                        if wrapper in result and isinstance(result[wrapper], (dict, list, str)):
                            candidate_map.update(_flatten_equivalences(result[wrapper]))
                    # if still empty, flatten top-level
                    if not candidate_map:
                        candidate_map.update(_flatten_equivalences(result))
                else:
                    candidate_map.update(_flatten_equivalences(result))

                if verbose:
                    self._log(f"   [LLM attempt {attempt+1}] Flattened mapping: {candidate_map}")

                # Filter and match to actual candidate skills with some fuzzy checks
                valid_matches: Dict[str, str] = {}
                for gap, match in candidate_map.items():
                    if not match or not isinstance(match, str):
                        continue
                    match_lower = match.lower()

                    # exact
                    if match_lower in candidate_lower:
                        valid_matches[gap] = candidate_lower[match_lower]
                        continue

                    # substring / token match (e.g., 'flask' vs 'flask-restful')
                    for cand_l, cand_orig in candidate_lower.items():
                        if cand_l in match_lower or match_lower in cand_l:
                            valid_matches[gap] = cand_orig
                            break

                if valid_matches:
                    return valid_matches

                if verbose:
                    self._log(f"   [LLM attempt {attempt+1}] No valid matches after flattening: {candidate_map}")

            except Exception as e:
                if verbose:
                    self._log(f"   [LLM attempt {attempt+1}] Error: {e}")
                continue

        return {}

    def _log(self, message: str) -> None:
        print(f"[LLMService] {message}")
    

