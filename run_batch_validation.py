import argparse
import csv
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

from src.orchestrator import MatchingOrchestrator
from src.services.esco_mapper import ESCOMapper
from src.services.llm_service import LLMService


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _model_to_dict(model: Any) -> Dict[str, Any]:
    if model is None:
        return {}
    if hasattr(model, "model_dump"):
        return model.model_dump()
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _iter_files(directory: Path, extensions: Tuple[str, ...]) -> List[Path]:
    paths: List[Path] = []
    for ext in extensions:
        paths.extend(directory.glob(f"*{ext}"))
    return sorted({p.resolve() for p in paths})


def _safe_skill_list(skills: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in skills or []:
        d = _model_to_dict(s)
        d.pop("raw_text", None)
        out.append(d)
    return out


def _safe_negotiation_log(negotiation_log: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for entry in negotiation_log or []:
        try:
            out.append(asdict(entry))
        except Exception:
            out.append(dict(entry))
    return out


def _existing_pairs(csv_path: Path) -> set[Tuple[str, str]]:
    if not csv_path.exists():
        return set()
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return {
                (row.get("cv_file", "") or "", row.get("jd_file", "") or "")
                for row in reader
                if row
            }
    except Exception:
        return set()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Esegue il matching per tutte le combinazioni CV x JD e salva un CSV "
            "con log/metriche utili per la validazione."
        )
    )
    parser.add_argument("--cvs-dir", default="data/sample_cvs", help="Directory contenente i CV (PDF o TXT).")
    parser.add_argument("--jds-dir", default="data/sample_jobs", help="Directory contenente le job description (TXT).")
    parser.add_argument("--out", default="data/validation/batch_results.csv", help="Percorso output CSV.")
    parser.add_argument("--limit", type=int, default=0, help="Se > 0, limita il numero di coppie processate.")
    parser.add_argument("--skip-existing", action="store_true", help="Salta coppie già presenti nel CSV output.")
    parser.add_argument("--verbose", action="store_true", help="Abilita log verbose dell'orchestrator.")

    parser.add_argument("--enable-refinement", action=argparse.BooleanOptionalAction, default=True, help="Abilita refinement loop (default: attivo). Usa --no-enable-refinement per disabilitare.")
    parser.add_argument("--refinement-threshold", type=float, default=50.0, help="Soglia score per attivare refinement.")
    parser.add_argument("--max-refinement-rounds", type=int, default=1, help="Massimo numero di round di refinement.")

    parser.add_argument("--llm-provider", choices=["ollama", "lmstudio"], default="lmstudio")
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--lmstudio-model", default=os.getenv("LMSTUDIO_MODEL", "meta/llama-3.3-70b"))
    parser.add_argument("--lmstudio-base-url", default=os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"))
    parser.add_argument("--lmstudio-api-key", default=os.getenv("LMSTUDIO_API_KEY", "lmstudio"))
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--timeout", type=int, default=120)

    parser.add_argument("--esco-csv", default="data/esco/skills_it.csv")
    parser.add_argument("--custom-tech-csv", default="data/custom_tech_skills.csv")
    parser.add_argument("--embedding-model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--esco-similarity-threshold", type=float, default=0.7)

    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent
    cvs_dir = (project_root / args.cvs_dir).resolve()
    jds_dir = (project_root / args.jds_dir).resolve()
    out_path = (project_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cv_paths = _iter_files(cvs_dir, extensions=(".pdf", ".txt"))
    jd_paths = _iter_files(jds_dir, extensions=(".txt",))

    if not cv_paths:
        raise SystemExit(f"Nessun CV trovato in: {cvs_dir}")
    if not jd_paths:
        raise SystemExit(f"Nessuna JD trovata in: {jds_dir}")

    existing = _existing_pairs(out_path) if args.skip_existing else set()

    llm_service = LLMService(
        provider=args.llm_provider,
        model=args.ollama_model,
        lmstudio_model=None if args.llm_provider == "ollama" else args.lmstudio_model,
        lmstudio_base_url=args.lmstudio_base_url,
        lmstudio_api_key=args.lmstudio_api_key,
        temperature=args.temperature,
        timeout=args.timeout,
    )

    esco_mapper = ESCOMapper(
        esco_csv_path=args.esco_csv,
        custom_csv_path=args.custom_tech_csv,
        model_name=args.embedding_model,
        similarity_threshold=args.esco_similarity_threshold,
    )

    orchestrator = MatchingOrchestrator(
        llm_service=llm_service,
        esco_mapper=esco_mapper,
        refinement_threshold=args.refinement_threshold,
        max_refinement_rounds=args.max_refinement_rounds,
        verbose=args.verbose,
    )

    fieldnames = [
        "run_id",
        "timestamp_utc",
        "cv_file",
        "jd_file",
        "enable_refinement",
        "refinement_threshold",
        "max_refinement_rounds",
        "llm_provider",
        "llm_model",
        "temperature",
        "timeout",
        "embedding_model",
        "esco_similarity_threshold",
        "job_title",
        "candidate_name",
        "candidate_experience",
        "required_experience",
        "n_required",
        "n_preferred",
        "n_candidate_skills",
        "initial_score",
        "final_score",
        "score_improvement",
        "required_score",
        "preferred_score",
        "experience_score",
        "total_rounds",
        "matched_count",
        "gaps_count",
        "n_match_esco_id",
        "n_match_esco_name",
        "n_match_original_name",
        "n_match_reverse",
        "n_match_fuzzy",
        "n_match_llm_reasoning",
        "matched_skills_json",
        "gaps_json",
        "strengths_json",
        "match_types_json",
        "explanation",
        "job_required_skills_json",
        "job_preferred_skills_json",
        "candidate_skills_json",
        "negotiation_log_json",
        "elapsed_ms",
        "error",
    ]

    write_header = not out_path.exists()
    processed = 0

    with out_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for jd_path in jd_paths:
            jd_text = _read_text(jd_path)
            for cv_path in cv_paths:
                if args.limit and processed >= args.limit:
                    return 0

                cv_name = cv_path.name
                jd_name = jd_path.name
                if (cv_name, jd_name) in existing:
                    continue

                run_id = f"{jd_path.stem}__{cv_path.stem}__{int(time.time())}"
                started = time.perf_counter()

                row: Dict[str, Any] = {
                    "run_id": run_id,
                    "timestamp_utc": _utc_now_iso(),
                    "cv_file": cv_name,
                    "jd_file": jd_name,
                    "enable_refinement": bool(args.enable_refinement),
                    "refinement_threshold": float(args.refinement_threshold),
                    "max_refinement_rounds": int(args.max_refinement_rounds),
                    "llm_provider": args.llm_provider,
                    "llm_model": llm_service.model,
                    "temperature": float(args.temperature),
                    "timeout": int(args.timeout),
                    "embedding_model": args.embedding_model,
                    "esco_similarity_threshold": float(args.esco_similarity_threshold),
                    "job_title": "",
                    "candidate_name": "",
                    "candidate_experience": "",
                    "required_experience": "",
                    "n_required": "",
                    "n_preferred": "",
                    "n_candidate_skills": "",
                    "initial_score": "",
                    "final_score": "",
                    "score_improvement": "",
                    "required_score": "",
                    "preferred_score": "",
                    "experience_score": "",
                    "total_rounds": "",
                    "matched_count": "",
                    "gaps_count": "",
                    "n_match_esco_id": "",
                    "n_match_esco_name": "",
                    "n_match_original_name": "",
                    "n_match_reverse": "",
                    "n_match_fuzzy": "",
                    "n_match_llm_reasoning": "",
                    "matched_skills_json": "",
                    "gaps_json": "",
                    "strengths_json": "",
                    "match_types_json": "",
                    "explanation": "",
                    "job_required_skills_json": "",
                    "job_preferred_skills_json": "",
                    "candidate_skills_json": "",
                    "negotiation_log_json": "",
                    "elapsed_ms": "",
                    "error": "",
                }

                try:
                    # Read text for non-PDF CVs (PDF handled by CandidateAgent)
                    cv_input = _read_text(cv_path) if cv_path.suffix.lower() != ".pdf" else str(cv_path)

                    result = orchestrator.run(
                        cv_input=cv_input,
                        job_description=jd_text,
                        enable_refinement=bool(args.enable_refinement),
                    )

                    job = result.job_requirements
                    candidate = result.candidate_profile
                    match = result.match_result

                    row["job_title"] = job.job_title or ""
                    row["candidate_name"] = candidate.name or ""
                    row["candidate_experience"] = candidate.experience_years or 0
                    row["required_experience"] = job.experience_years or 0

                    row["n_required"] = len(job.required_skills or [])
                    row["n_preferred"] = len(job.preferred_skills or [])
                    row["n_candidate_skills"] = len(candidate.skills or [])

                    row["initial_score"] = float(getattr(result, "initial_score", match.score))
                    row["final_score"] = float(getattr(result, "final_score", match.score))
                    row["score_improvement"] = float(getattr(result, "score_improvement", 0.0))
                    row["required_score"] = float(match.required_score)
                    row["preferred_score"] = float(match.preferred_score)
                    row["experience_score"] = float(match.experience_score)
                    row["total_rounds"] = int(getattr(result, "total_rounds", 1))

                    row["matched_count"] = len(match.matched_skills or [])
                    row["gaps_count"] = len(match.gaps or [])

                    # Breakdown match per tipo
                    mt = match.match_types or {}
                    row["n_match_esco_id"] = sum(1 for v in mt.values() if v == "esco_id")
                    row["n_match_esco_name"] = sum(1 for v in mt.values() if v == "esco_name")
                    row["n_match_original_name"] = sum(1 for v in mt.values() if v == "original_name")
                    row["n_match_reverse"] = sum(1 for v in mt.values() if v == "reverse")
                    row["n_match_fuzzy"] = sum(1 for v in mt.values() if v == "fuzzy")
                    row["n_match_llm_reasoning"] = sum(1 for v in mt.values() if v == "llm_reasoning")

                    row["matched_skills_json"] = _json_dumps(match.matched_skills or [])
                    row["gaps_json"] = _json_dumps(match.gaps or [])
                    row["strengths_json"] = _json_dumps(match.strengths or [])
                    row["match_types_json"] = _json_dumps(mt)
                    row["explanation"] = match.explanation or ""

                    row["job_required_skills_json"] = _json_dumps(_safe_skill_list(job.required_skills))
                    row["job_preferred_skills_json"] = _json_dumps(_safe_skill_list(job.preferred_skills))
                    row["candidate_skills_json"] = _json_dumps(_safe_skill_list(candidate.skills))
                    row["negotiation_log_json"] = _json_dumps(_safe_negotiation_log(result.negotiation_log))

                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"
                finally:
                    row["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
                    writer.writerow(row)
                    f.flush()
                    processed += 1
                    pair_label = f"{cv_name} x {jd_name}"
                    if row["error"]:
                        print(f"  [{processed}] ERRORE {pair_label}: {row['error']}")
                    else:
                        print(f"  [{processed}] OK {pair_label} -> score={row['final_score']}")

    # ═══════════════════════════════════════════════════════════════
    # Summary statistics for validation chapter
    # ═══════════════════════════════════════════════════════════════
    stats_path = out_path.parent / f"{out_path.stem}_stats.csv"
    _generate_validation_stats(out_path, stats_path)

    return 0


# ═══════════════════════════════════════════════════════════════════════
# VALIDATION STATS GENERATION
# ═══════════════════════════════════════════════════════════════════════

def _safe_float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row[key])
    except (ValueError, KeyError, TypeError):
        return default


def _safe_int(row: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(row[key])
    except (ValueError, KeyError, TypeError):
        return default


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def _std_dev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / (len(values) - 1)) ** 0.5


def _generate_validation_stats(csv_path: Path, stats_path: Path) -> None:
    """Genera un CSV di statistiche per il capitolo di validazione della tesi."""
    if not csv_path.exists():
        return
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return

    if not rows:
        print("\nNessun risultato nel CSV.")
        return

    ok_rows = [r for r in rows if not r.get("error")]
    error_rows = [r for r in rows if r.get("error")]

    # ── Raccogli metriche base ──
    scores = [_safe_float(r, "final_score") for r in ok_rows]
    initial_scores = [_safe_float(r, "initial_score") for r in ok_rows]
    improvements = [_safe_float(r, "score_improvement") for r in ok_rows]
    required_scores = [_safe_float(r, "required_score") for r in ok_rows]
    preferred_scores = [_safe_float(r, "preferred_score") for r in ok_rows]
    experience_scores = [_safe_float(r, "experience_score") for r in ok_rows]
    elapsed_list = [_safe_int(r, "elapsed_ms") for r in ok_rows]
    matched_list = [_safe_int(r, "matched_count") for r in ok_rows]
    gaps_list = [_safe_int(r, "gaps_count") for r in ok_rows]
    n_required_list = [_safe_int(r, "n_required") for r in ok_rows]
    n_preferred_list = [_safe_int(r, "n_preferred") for r in ok_rows]
    n_candidate_list = [_safe_int(r, "n_candidate_skills") for r in ok_rows]

    # ── Match type counts ──
    match_type_keys = [
        "n_match_esco_id", "n_match_esco_name", "n_match_original_name",
        "n_match_reverse", "n_match_fuzzy", "n_match_llm_reasoning",
    ]
    match_type_totals = {k: sum(_safe_int(r, k) for r in ok_rows) for k in match_type_keys}
    total_matches_all = sum(match_type_totals.values())

    # ── Score distribution ──
    buckets = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for s in scores:
        if s <= 20:
            buckets["0-20"] += 1
        elif s <= 40:
            buckets["21-40"] += 1
        elif s <= 60:
            buckets["41-60"] += 1
        elif s <= 80:
            buckets["61-80"] += 1
        else:
            buckets["81-100"] += 1

    # ── Refinement effectiveness ──
    refined = [r for r in ok_rows if _safe_float(r, "score_improvement") > 0]
    non_zero_imp = [_safe_float(r, "score_improvement") for r in refined]

    # ── Per-JD aggregates ──
    jd_scores: Dict[str, List[float]] = {}
    for r in ok_rows:
        jd = r.get("jd_file", "")
        jd_scores.setdefault(jd, []).append(_safe_float(r, "final_score"))

    # ── Per-CV aggregates ──
    cv_scores: Dict[str, List[float]] = {}
    for r in ok_rows:
        cv = r.get("cv_file", "")
        cv_scores.setdefault(cv, []).append(_safe_float(r, "final_score"))

    # ═══════════════ Build stats rows ═══════════════
    stats: List[Dict[str, str]] = []

    def _add(section: str, metric: str, value: Any) -> None:
        stats.append({"section": section, "metric": metric, "value": str(value)})

    # — Overview —
    _add("overview", "total_pairs", len(rows))
    _add("overview", "ok_pairs", len(ok_rows))
    _add("overview", "error_pairs", len(error_rows))
    _add("overview", "error_rate_%", f"{len(error_rows) / len(rows) * 100:.1f}" if rows else "0")

    # — Score distribution —
    if scores:
        _add("score", "mean", f"{sum(scores)/len(scores):.2f}")
        _add("score", "median", f"{_median(scores):.2f}")
        _add("score", "std_dev", f"{_std_dev(scores):.2f}")
        _add("score", "min", f"{min(scores):.2f}")
        _add("score", "max", f"{max(scores):.2f}")
        for bucket, count in buckets.items():
            _add("score_distribution", f"bucket_{bucket}", count)

    # — Score breakdown —
    if required_scores:
        _add("score_breakdown", "required_mean", f"{sum(required_scores)/len(required_scores):.2f}")
        _add("score_breakdown", "required_median", f"{_median(required_scores):.2f}")
    if preferred_scores:
        _add("score_breakdown", "preferred_mean", f"{sum(preferred_scores)/len(preferred_scores):.2f}")
        _add("score_breakdown", "preferred_median", f"{_median(preferred_scores):.2f}")
    if experience_scores:
        _add("score_breakdown", "experience_mean", f"{sum(experience_scores)/len(experience_scores):.2f}")
        _add("score_breakdown", "experience_median", f"{_median(experience_scores):.2f}")

    # — Match types —
    for k, v in match_type_totals.items():
        label = k.replace("n_match_", "")
        _add("match_type", f"total_{label}", v)
        pct = f"{v / total_matches_all * 100:.1f}" if total_matches_all else "0"
        _add("match_type", f"pct_{label}", pct)

    # — Skills —
    if matched_list:
        _add("skills", "matched_mean", f"{sum(matched_list)/len(matched_list):.2f}")
    if gaps_list:
        _add("skills", "gaps_mean", f"{sum(gaps_list)/len(gaps_list):.2f}")
    if n_required_list:
        _add("skills", "required_mean", f"{sum(n_required_list)/len(n_required_list):.2f}")
    if n_preferred_list:
        _add("skills", "preferred_mean", f"{sum(n_preferred_list)/len(n_preferred_list):.2f}")
    if n_candidate_list:
        _add("skills", "candidate_mean", f"{sum(n_candidate_list)/len(n_candidate_list):.2f}")

    # — Refinement —
    _add("refinement", "pairs_improved", len(refined))
    _add("refinement", "pairs_not_improved", len(ok_rows) - len(refined))
    if non_zero_imp:
        _add("refinement", "avg_improvement", f"{sum(non_zero_imp)/len(non_zero_imp):.2f}")
        _add("refinement", "max_improvement", f"{max(non_zero_imp):.2f}")
    if initial_scores and scores:
        _add("refinement", "initial_score_mean", f"{sum(initial_scores)/len(initial_scores):.2f}")

    # — Timing —
    if elapsed_list:
        _add("timing", "mean_ms", f"{sum(elapsed_list)/len(elapsed_list):.0f}")
        _add("timing", "median_ms", f"{_median([float(e) for e in elapsed_list]):.0f}")
        _add("timing", "min_ms", str(min(elapsed_list)))
        _add("timing", "max_ms", str(max(elapsed_list)))
        _add("timing", "total_s", f"{sum(elapsed_list)/1000:.1f}")

    # — Per-JD breakdown —
    for jd_name, jd_s in sorted(jd_scores.items()):
        _add("per_jd", f"{jd_name}|mean", f"{sum(jd_s)/len(jd_s):.2f}")
        _add("per_jd", f"{jd_name}|median", f"{_median(jd_s):.2f}")
        _add("per_jd", f"{jd_name}|min", f"{min(jd_s):.2f}")
        _add("per_jd", f"{jd_name}|max", f"{max(jd_s):.2f}")
        _add("per_jd", f"{jd_name}|count", str(len(jd_s)))

    # — Per-CV breakdown —
    for cv_name, cv_s in sorted(cv_scores.items()):
        _add("per_cv", f"{cv_name}|mean", f"{sum(cv_s)/len(cv_s):.2f}")
        _add("per_cv", f"{cv_name}|min", f"{min(cv_s):.2f}")
        _add("per_cv", f"{cv_name}|max", f"{max(cv_s):.2f}")

    # — Config (from first row) —
    if rows:
        r0 = rows[0]
        for key in ("llm_provider", "llm_model", "temperature", "timeout",
                     "embedding_model", "esco_similarity_threshold",
                     "enable_refinement", "refinement_threshold", "max_refinement_rounds"):
            _add("config", key, r0.get(key, ""))

    # ═══════════════ Write stats CSV ═══════════════
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w", encoding="utf-8", newline="") as sf:
        w = csv.DictWriter(sf, fieldnames=["section", "metric", "value"])
        w.writeheader()
        w.writerows(stats)

    # ═══════════════ Print to console ═══════════════
    print("\n" + "=" * 70)
    print("  SUMMARY – Batch Validation Results")
    print("=" * 70)
    print(f"  Coppie: {len(rows)} totali ({len(ok_rows)} OK, {len(error_rows)} errori)")
    if scores:
        print(f"  Score: media={sum(scores)/len(scores):.1f}  mediana={_median(scores):.1f}  "
              f"min={min(scores):.1f}  max={max(scores):.1f}  std={_std_dev(scores):.1f}")
        print(f"  Distribuzione: {buckets}")
    if required_scores:
        print(f"  Breakdown: required={sum(required_scores)/len(required_scores):.1f}  "
              f"preferred={sum(preferred_scores)/len(preferred_scores):.1f}  "
              f"experience={sum(experience_scores)/len(experience_scores):.1f}")
    if total_matches_all:
        parts = [f"{k.replace('n_match_', '')}={v}" for k, v in match_type_totals.items() if v]
        print(f"  Match types ({total_matches_all} tot): {', '.join(parts)}")
    if non_zero_imp:
        print(f"  Refinement: {len(refined)}/{len(ok_rows)} migliorati, avg +{sum(non_zero_imp)/len(non_zero_imp):.1f}")
    if elapsed_list:
        print(f"  Tempo: media={sum(elapsed_list)/len(elapsed_list):.0f}ms  tot={sum(elapsed_list)/1000:.1f}s")
    print(f"  Output CSV: {csv_path}")
    print(f"  Stats CSV:  {stats_path}")
    print("=" * 70)


if __name__ == "__main__":
    raise SystemExit(main())

