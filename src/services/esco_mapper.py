"""
ESCO Mapper Service (Hybrid)
Mappa skill estratte usando:
1. Custom Tech Skills (priorità alta) - per skill tech moderne
2. ESCO (fallback) - per skill generiche

ID Format:
- custom:tech/react  → Skill tech custom
- http://data.europa.eu/esco/skill/...  → ESCO ufficiale

Ottimizzazione:
- Cache embeddings su disco (.npy) per caricamento istantaneo
- Supporto multilingua (it/en)
"""

import pandas as pd
import numpy as np
import hashlib
from sentence_transformers import SentenceTransformer
from typing import Optional, Tuple, List
from pathlib import Path


class ESCOMapper:
    """
    Servizio ibrido per mappare skill free-text.
    
    Strategia:
    1. Prima cerca match esatto/alias nel dizionario tech custom
    2. Se non trova, usa embedding similarity su ESCO
    
    Questo garantisce match precisi per skill tech moderne (React, Docker, etc.)
    mantenendo la copertura ESCO per skill più generiche.
    """
    
    def __init__(
        self,
        esco_csv_path: str = "data/esco/skills_it.csv",  # Default: italiano
        custom_csv_path: str = "data/custom_tech_skills.csv",
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold: float = 0.5,
        use_custom_skills: bool = True,
        cache_dir: str = "data/cache" 
    ):
        """
        Inizializza l'ESCO Mapper ibrido.
        
        Args:
            esco_csv_path: Percorso al CSV con le skill ESCO (default: italiano)
            custom_csv_path: Percorso al CSV con skill tech custom
            model_name: Nome del modello sentence-transformers
            similarity_threshold: Soglia minima di similarità per match ESCO
            use_custom_skills: Se True, cerca prima nel dizionario custom
            cache_dir: Directory per cache embeddings (velocizza riavvii)
        """
        self.similarity_threshold = similarity_threshold
        self.use_custom_skills = use_custom_skills
        self.cache_dir = Path(cache_dir) if not Path(cache_dir).is_absolute() else Path(cache_dir)
        
        # Resolve project root
        self.project_root = Path(__file__).parent.parent.parent
        if not self.cache_dir.is_absolute():
            self.cache_dir = self.project_root / cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Carica skill tech custom (lookup veloce)
        if use_custom_skills:
            self._log("Caricamento skill tech custom...")
            self._load_custom_skills(custom_csv_path)
        
        # Carica il modello di embedding
        self._log(f"Caricamento modello: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        
        # Pre-calcola embeddings per custom skills (per fallback semantico)
        if use_custom_skills:
            self._compute_custom_embeddings()
        
        # Carica il dataset ESCO
        self._log(f"Caricamento dataset ESCO da: {esco_csv_path}...")
        self._load_esco_data(esco_csv_path)
        
        # Pre-calcola o carica da cache gli embeddings delle skill ESCO
        self._load_or_compute_embeddings(esco_csv_path)
        
        self._log("ESCO Mapper pronto")
    
    def _load_custom_skills(self, csv_path: str) -> None:
        """Carica il dizionario di skill tech custom."""
        if not Path(csv_path).is_absolute():
            project_root = Path(__file__).parent.parent.parent
            csv_path = project_root / csv_path
        
        self.custom_df = pd.read_csv(csv_path)
        
        # Crea lookup dictionary: nome/alias (lowercase) → (skill_id, nome_ufficiale)
        self.custom_lookup = {}
        
        for _, row in self.custom_df.iterrows():
            skill_id = row['skill_id']
            name = row['name']
            
            # Aggiungi nome principale
            self.custom_lookup[name.lower()] = (skill_id, name, row['category'])
            
            # Aggiungi alias
            if pd.notna(row['aliases']) and row['aliases']:
                for alias in row['aliases'].split(','):
                    alias = alias.strip()
                    if alias:
                        self.custom_lookup[alias.lower()] = (skill_id, name, row['category'])
        
        # Prepara lista nomi per embeddings (fallback semantico)
        self.custom_names = self.custom_df['name'].tolist()
        self.custom_ids = self.custom_df['skill_id'].tolist()
        self.custom_categories = self.custom_df['category'].tolist()
        
        self._log(f"   -> {len(self.custom_df)} skill tech, {len(self.custom_lookup)} lookup entries")
    
    def _compute_custom_embeddings(self) -> None:
        """Pre-calcola embeddings per skill custom (fallback semantico)."""
        if hasattr(self, 'custom_names') and self.custom_names:
            self.custom_embeddings = self.model.encode(
                self.custom_names,
                convert_to_numpy=True
            )
            self.custom_embeddings = self.custom_embeddings / np.linalg.norm(
                self.custom_embeddings, axis=1, keepdims=True
            )
    
    def _load_esco_data(self, csv_path: str) -> None:
        """Carica e preprocessa il dataset ESCO."""
        # Trova il percorso assoluto
        if not Path(csv_path).is_absolute():
            # Cerca dalla root del progetto
            project_root = Path(__file__).parent.parent.parent
            csv_path = project_root / csv_path
        
        self.esco_df = pd.read_csv(csv_path)
        
        # Estrai le colonne rilevanti
        self.esco_skills = self.esco_df[[
            'conceptUri',      # esco_id
            'preferredLabel',  # nome principale
            'altLabels',       # nomi alternativi
            'description'      # descrizione (utile per embedding più ricco)
        ]].copy()
        
        # Pulisci i dati
        self.esco_skills['preferredLabel'] = self.esco_skills['preferredLabel'].fillna('')
        self.esco_skills['altLabels'] = self.esco_skills['altLabels'].fillna('')
        self.esco_skills['description'] = self.esco_skills['description'].fillna('')
        
        # Crea un testo combinato per embedding più ricco
        # (nome + alternative danno più contesto semantico)
        self.esco_skills['combined_text'] = (
            self.esco_skills['preferredLabel'] + ' ' + 
            self.esco_skills['altLabels'].str.replace('\n', ' ')
        ).str.strip()
    
    def _get_cache_path(self, csv_path: str) -> Path:
        """Genera path univoco per cache basato su CSV + modello."""
        # Hash basato su: nome file CSV + modello + numero righe
        csv_name = Path(csv_path).stem  # es. "skills_it"
        cache_key = f"{csv_name}_{self.model_name}_{len(self.esco_skills)}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        return self.cache_dir / f"esco_embeddings_{csv_name}_{cache_hash}.npy"
    
    def _load_or_compute_embeddings(self, csv_path: str) -> None:
        """Carica embeddings da cache o calcola e salva."""
        cache_path = self._get_cache_path(csv_path)
        
        if cache_path.exists():
            self._log(f"Caricamento embeddings da cache: {cache_path.name}")
            self.esco_embeddings = np.load(cache_path)
            self._log(f"   -> {self.esco_embeddings.shape[0]} embeddings caricati")
        else:
            self._log(f"Calcolo embeddings per {len(self.esco_skills)} skill ESCO...")
            self._log("   (prima volta, verra salvato in cache)")
            self._compute_esco_embeddings()
            
            # Salva in cache
            np.save(cache_path, self.esco_embeddings)
            self._log(f"Cache salvata: {cache_path.name}")
    
    def _compute_esco_embeddings(self) -> None:
        """Pre-calcola gli embeddings per tutte le skill ESCO."""
        texts = self.esco_skills['combined_text'].tolist()
        self.esco_embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        # Normalizza per cosine similarity efficiente
        self.esco_embeddings = self.esco_embeddings / np.linalg.norm(
            self.esco_embeddings, axis=1, keepdims=True
        )
    
    def map_skill(self, skill_name: str) -> Tuple[Optional[str], Optional[str], float, str]:
        """
        Mappa una skill free-text (ibrido: custom first, then ESCO).
        
        Strategia a 4 livelli:
        1. Match esatto custom (lookup)
        2. Match con nome base (prima delle parentesi)
        3. Similarità semantica su custom (embeddings)
        4. Similarità semantica su ESCO (fallback)
        
        Args:
            skill_name: Nome della skill da mappare (es. "React", "Python")
            
        Returns:
            Tupla (skill_id, normalized_name, confidence, source):
                - skill_id: ID della skill (custom:... o http://...esco...)
                - normalized_name: Nome normalizzato della skill
                - confidence: Score di similarità (0-1), 1.0 per match esatto custom
                - source: "custom" o "esco" per indicare la fonte
        """
        if not skill_name or not skill_name.strip():
            return None, None, 0.0, "none"
        
        skill_lower = skill_name.lower().strip()
        
        # STEP 1: Cerca match esatto nel dizionario custom
        if self.use_custom_skills and skill_lower in self.custom_lookup:
            skill_id, normalized_name, category = self.custom_lookup[skill_lower]
            return skill_id, normalized_name, 1.0, "custom"
        
        # STEP 2: Se contiene parentesi, prova con il nome base
        # Es: "AWS (EC2, S3, Lambda)" → prova "AWS"
        if '(' in skill_name:
            base_name = skill_name.split('(')[0].strip().lower()
            if self.use_custom_skills and base_name in self.custom_lookup:
                skill_id, normalized_name, category = self.custom_lookup[base_name]
                return skill_id, normalized_name, 1.0, "custom"
        
        # Calcola embedding per la skill da cercare
        skill_embedding = self.model.encode(skill_name, convert_to_numpy=True)
        skill_embedding = skill_embedding / np.linalg.norm(skill_embedding)
        
        # STEP 3: Similarità semantica su custom tech skills
        # (priorità su ESCO perché più rilevanti per tech)
        if self.use_custom_skills and hasattr(self, 'custom_embeddings'):
            custom_similarities = np.dot(self.custom_embeddings, skill_embedding)
            best_custom_idx = custom_similarities.argmax()
            best_custom_sim = custom_similarities[best_custom_idx]
            
            # Soglia più alta per custom (0.8) per evitare match troppo generici
            if best_custom_sim >= 0.8:
                return (
                    self.custom_ids[best_custom_idx],
                    self.custom_names[best_custom_idx],
                    float(best_custom_sim),
                    "custom"
                )
        
        # STEP 4: Fallback a ESCO via embeddings
        similarities = np.dot(self.esco_embeddings, skill_embedding)
        best_idx = similarities.argmax()
        best_similarity = similarities[best_idx]
        
        if best_similarity < self.similarity_threshold:
            return None, None, float(best_similarity), "none"
        
        best_match = self.esco_skills.iloc[best_idx]
        return (
            best_match['conceptUri'],
            best_match['preferredLabel'],
            float(best_similarity),
            "esco"
        )
    
    def map_skills_batch(
        self, 
        skill_names: List[str]
    ) -> List[Tuple[Optional[str], Optional[str], float, str]]:
        """
        Mappa un batch di skill (più efficiente per molte skill).
        
        Args:
            skill_names: Lista di nomi skill da mappare
            
        Returns:
            Lista di tuple (skill_id, normalized_name, confidence, source)
        """
        if not skill_names:
            return []
        
        results = []
        esco_pending = []  # (original_idx, skill_name)
        
        # Prima passa: match esatto custom
        for i, skill_name in enumerate(skill_names):
            if not skill_name or not skill_name.strip():
                results.append((None, None, 0.0, "none"))
                continue
            
            skill_lower = skill_name.lower().strip()
            
            # Match esatto
            if self.use_custom_skills and skill_lower in self.custom_lookup:
                skill_id, normalized_name, category = self.custom_lookup[skill_lower]
                results.append((skill_id, normalized_name, 1.0, "custom"))
            # Match con nome base (prima delle parentesi)
            elif self.use_custom_skills and '(' in skill_name:
                base_name = skill_name.split('(')[0].strip().lower()
                if base_name in self.custom_lookup:
                    skill_id, normalized_name, category = self.custom_lookup[base_name]
                    results.append((skill_id, normalized_name, 1.0, "custom"))
                else:
                    results.append(None)  # Placeholder
                    esco_pending.append((i, skill_name))
            else:
                results.append(None)  # Placeholder
                esco_pending.append((i, skill_name))
        
        # Seconda passa: embeddings per quelli non trovati
        if esco_pending:
            indices, texts = zip(*esco_pending)
            skill_embeddings = self.model.encode(
                list(texts),
                convert_to_numpy=True,
                show_progress_bar=len(texts) > 10
            )
            skill_embeddings = skill_embeddings / np.linalg.norm(
                skill_embeddings, axis=1, keepdims=True
            )
            
            # Step 2b: Prima prova custom embeddings (soglia 0.8)
            if self.use_custom_skills and hasattr(self, 'custom_embeddings'):
                custom_sims = np.dot(skill_embeddings, self.custom_embeddings.T)
                
                for i, orig_idx in enumerate(indices):
                    best_custom_idx = custom_sims[i].argmax()
                    best_custom_sim = custom_sims[i][best_custom_idx]
                    
                    if best_custom_sim >= 0.8:
                        results[orig_idx] = (
                            self.custom_ids[best_custom_idx],
                            self.custom_names[best_custom_idx],
                            float(best_custom_sim),
                            "custom"
                        )
            
            # Step 3: ESCO embeddings per quelli ancora non trovati
            esco_sims = np.dot(skill_embeddings, self.esco_embeddings.T)
            
            for i, orig_idx in enumerate(indices):
                # Salta se già trovato con custom
                if results[orig_idx] is not None:
                    continue
                    
                best_idx = esco_sims[i].argmax()
                best_similarity = esco_sims[i][best_idx]
                
                if best_similarity >= self.similarity_threshold:
                    best_match = self.esco_skills.iloc[best_idx]
                    results[orig_idx] = (
                        best_match['conceptUri'],
                        best_match['preferredLabel'],
                        float(best_similarity),
                        "esco"
                    )
                else:
                    results[orig_idx] = (None, None, float(best_similarity), "none")
        
        return results
    
    def get_top_matches(
        self, 
        skill_name: str, 
        top_k: int = 5
    ) -> List[Tuple[str, str, float]]:
        """
        Ritorna i top-k match per una skill (utile per debug/analisi).
        
        Args:
            skill_name: Nome della skill
            top_k: Numero di match da ritornare
            
        Returns:
            Lista di tuple (esco_id, esco_name, confidence) ordinate per similarità
        """
        if not skill_name or not skill_name.strip():
            return []
        
        # Genera embedding
        skill_embedding = self.model.encode(skill_name, convert_to_numpy=True)
        skill_embedding = skill_embedding / np.linalg.norm(skill_embedding)
        
        # Calcola similarità
        similarities = np.dot(self.esco_embeddings, skill_embedding)
        
        # Trova top-k
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            match = self.esco_skills.iloc[idx]
            results.append((
                match['conceptUri'],
                match['preferredLabel'],
                float(similarities[idx])
            ))
        
        return results

    def _log(self, message: str) -> None:
        print(f"[ESCOMapper] {message}")
