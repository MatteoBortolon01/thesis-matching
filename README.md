# Sistema Multi-Agente per Job Matching

Sistema di matching CV-Job basato su architettura multi-agente con integrazione ESCO.

> **Tesi Universitaria** - Stack 100% gratuito e locale

---

## 📋 Indice

1. [Panoramica](#-panoramica)
2. [Architettura](#-architettura)
3. [Stack Tecnologico](#-stack-tecnologico)
4. [Struttura Progetto](#-struttura-progetto)
5. [Setup](#-setup)
6. [Flusso di Esecuzione](#-flusso-di-esecuzione)
7. [Agenti](#-agenti)
8. [ESCO Integration](#-esco-integration)

---

## Panoramica

### Obiettivi del Prototipo
- ✅ Parsing automatico di CV (PDF)
- ✅ Estrazione skill con LLM (CV e JD)
- ✅ Normalizzazione skill tramite tassonomia ESCO + Custom Tech Skills
- ✅ Matching intelligente con spiegazione LLM
- ✅ Demo interattiva Streamlit


---

## Architettura

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI                             │
│                   (Upload CV, Inserisci Job)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATOR                               │
│              (Coordina il flusso degli agenti)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  JOB AGENT    │    │  CANDIDATE    │    │   MATCHING    │
│               │    │    AGENT      │    │    AGENT      │
│ • Analizza JD │    │ • Parsa CV    │    │ • Negozia     │
│ • Estrae req  │    │ • Estrae skill│    │ • Calcola fit │
│ • Pesi skill  │    │ • Confidence  │    │ • Spiega      │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────┬───────────────────┬───────────────────────┐
│   CV Parser / LLM   │   ESCO Mapper     │   LLM Service         │
│   (LLM-based)       │   (embeddings +   │   (Ollama wrapper)    │
│                     │    numpy)         │                       │
└─────────────────────┴───────────────────┴───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OLLAMA + LLAMA 3.2 (3B)                      │
│              Estrazione requisiti + Spiegazioni                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack Tecnologico (100% Gratuito)

| Componente | Tecnologia | Motivo |
|------------|------------|--------|
| **Architettura** | Custom Python Classes | Agenti implementati come classi OOP |
| **LLM** | Ollama + Llama 3.2 (3B) | Locale, estrazione requisiti + spiegazioni (CV & JD) |
| **Tassonomia** | ESCO (subset IT) | Normalizzazione skill (~500 più comuni) |
| **Embeddings** | sentence-transformers | Cosine similarity in memoria (numpy) |
| **Frontend** | Streamlit | Demo interattiva |
| **Data Models** | Pydantic | Validazione e schema |

---

## 📁 Struttura Progetto

```
thesis-matching/
│
│
├── data/
│   ├── esco/                  # Dataset ESCO
│   │   └── skills_it.csv   
│   ├── sample_cvs/            # CV di test (PDF)
│   └── sample_jobs/           # Job description di test
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/                # DATA MODELS (Pydantic)
│   │   ├── __init__.py
│   │   ├── skill.py           # Skill + ESCO mapping
│   │   ├── job.py             # JobRequirements
│   │   ├── candidate.py       # CandidateProfile
│   │   └── match_result.py    # MatchResult (score, gaps, explanation)
│   │
│   ├── services/              # SERVIZI
│   │   ├── __init__.py
│   │   ├── esco_mapper.py     # Embeddings → ESCO (numpy)
│   │   └── llm_service.py     # Wrapper Ollama
│   │
│   ├── agents/                # AGENTI (Custom Classes)
│   │   ├── __init__.py
│   │   ├── job_agent.py       # Analizza JD con LLM
│   │   ├── candidate_agent.py # Analizza CV con LLM
│   │   └── matching_agent.py  # Calcolo score + spiegazione
│   │
│   └── orchestrator/          # COORDINAMENTO
│       ├── __init__.py
│       └── matching_orchestrator.py  # Coordina i 3 agenti
│
└── app/
    └── streamlit_app.py       # Demo (upload CV + input JD)

```

---

## 🚀 Setup

### 1. Prerequisiti

```bash
# Python 3.10+
python --version

# Ollama (per LLM locale)
# Scarica da: https://ollama.ai
ollama --version
```

### 2. Installa Ollama e modello

```bash
# Scarica Llama 3.2 (3B - leggero)
**Hardware minimo**:
- RAM: 8 GB (16 GB consigliato)
- Disk: 10 GB liberi
- CPU: qualsiasi (GPU opzionale)

**Software**:
```powershell
# Python 3.10+
python --version

# Ollama (scarica da https://ollama.ai)
ollama --version
```

### 2. Installa Ollama e modello

```powershell
# Scarica Llama 3.2 3B (leggero, perfetto per tesi)
ollama pull llama3.2
```

### 3. Setup Python

```powershell
# Crea virtual environment
python -m venv venv

# Attiva (Windows PowerShell)
.\venv\Scripts\activate

# Installa dipendenze (solo 8 librerie!)
pip install -r requirements.txt
```

### 4. Prepara dati ESCO

```powershell
# Scarica da: https://esco.ec.europa.eu/en/use-esco/download
```

### 5. Avvia Demo

```powershell
streamlit run app/streamlit_app.py
```

Apri browser: `http://localhost:8501 └── Job Parser estrae: requisiti, skill, pesi
                                        ▼
3. NORMALIZZAZIONE ESCO
   └── Skill estratte → mapping ESCO via embeddings
                                        ▼
4. AGENT NEGOTIATION
   ├── Job Agent: "Cerco Python, SQL"
   ├── Candidate Agent: "Ho Python , JavaScript"
   └── Matching Agent: media, negozia, decide
                                        ▼
5. OUTPUT
   ├── Match Score (0-100)
   ├── Breakdown per skill
   ├── Gap Analysis
   └── Spiegazione in linguaggio naturale
```

---

## Agenti

### Job Agent
**Ruolo**: Rappresenta gli interessi dell'azienda

**Input**: Job description (testo)

**Output**:
```python
{
    "required_skills": [
        {"skill": "Python", "esco_uri": "...", "weight": 0.9, "level": "advanced"}
    ],
    "preferred_skills": [...],
    "context": {
        "seniority": "mid",
        "sector": "fintech",
        "remote": True
    }
}
```

**Comportamento**:
- Estrae skill indispensabili vs preferenziali
- Assegna pesi in base al contesto
- Negozia: "Posso accettare junior se ha skill X"

---

### Candidate Agent
**Ruolo**: Rappresenta il candidato e valorizza il suo profilo

**Input**: CV (PDF/testo parsato)

**Output**:
```python
{
    "skills": [
        {"skill": "Python", "esco_uri": "...", "confidence": 0.95, "years": 3}
    ],
    "experience": [...],
    "languages": [
        {"language": "English", "level": "B2"}
    ],
    "projects": [...]
}
```

**Comportamento**:
- Estrae e valorizza skill anche implicite
- Calcola confidence basata su evidenze nel CV
- Negozia: "Non ho SQL ma ho PostgreSQL che è correlato"

---

### Matching Agent
**Ruolo**: Mediatore imparziale, decide il match

**Input**: Output di Job Agent + Candidate Agent

**Output**:
```python
{
    "match_score": 78,
    "breakdown": {
        "required_skills_match": 85,
        "preferred_skills_match": 60,
        "experience_match": 80,
        "language_match": 90
    },
    "gaps": ["Manca esperienza con Docker"],
    "strengths": ["Ottima conoscenza Python", "Progetti rilevanti"],
    "explanation": "Il candidato è un buon match per la posizione..."
}
```

**Comportamento**:
- Riceve "proposte" dai due agenti
- Applica logica di matching pesata
- Genera spiegazione human-readable

---

## 🇪🇺 ESCO Integration

### Cos'è ESCO?
- **European Skills, Competences, Qualifications and Occupations**
- ~14.000 skill standardizzate
- ~3.000 occupazioni
- Multilingue (italiano incluso)
- Relazioni semantiche tra skill

---

*Ultimo aggiornamento: Gennaio 2026*
