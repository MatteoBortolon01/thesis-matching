# Sistema Multi-Agente per Job Matching

Sistema di matching CV-Job basato su architettura multi-agente con integrazione ESCO.


## Indice

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
- Parsing automatico di CV (PDF)
- Estrazione skill con LLM (CV e JD)
- Normalizzazione skill tramite tassonomia ESCO + Custom Tech Skills
- Matching intelligente con spiegazione LLM

---

## Architettura

```
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
│   (LLM-based)       │   (embeddings +   │                       │
│                     │    numpy)         │                       │
└─────────────────────┴───────────────────┴───────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                            LLM                                  │
│              Estrazione requisiti + Spiegazioni                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack Tecnologico 

| Componente | Tecnologia | Motivo |
|------------|------------|--------|
| **Architettura** | Custom Python Classes | Agenti implementati come classi OOP |
| **LLM** | Ollama | Locale, estrazione requisiti + spiegazioni (CV & JD) |
| **Tassonomia** | ESCO | Normalizzazione skill (~500 più comuni) |
| **Embeddings** | sentence-transformers | Cosine similarity in memoria (numpy) |
| **Data Models** | Pydantic | Validazione e schema |

---

## Struttura Progetto

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
│   │   └── llm_service.py     # Wrapper LLM
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

```

### 1. Setup Python

```powershell
# Crea virtual environment
python -m venv venv

# Attiva (Windows PowerShell)
.\venv\Scripts\activate

# Installa dipendenze (solo 8 librerie!)
pip install -r requirements.txt
```

### 2. Prepara dati ESCO

```powershell
# Scarica da: https://esco.ec.europa.eu/en/use-esco/download
```

