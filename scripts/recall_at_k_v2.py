"""v2 evaluation -- local recall@k / MRR / complete-partial-miss diagnostic
for the frozen 40-question benchmark (eval/frozen_retrieval_eval_v2_40.json)
against the large (263-chunk, 9-chapter) corpus. Zero API calls.

Modeled on scripts/recall_at_k_large.py (same arm definitions, same
RETRIEVAL config, same _stored_label_covers containment logic), extended to:

  1. Score against the v2 40-question set instead of the frozen 15.
  2. For multi-clause questions (2+ distinct source_clauses), additionally
     report complete-evidence recall (ALL required clauses present in the
     top-k pool) vs. partial (SOME present) vs. miss (NONE present), per the
     Day-of-freeze plan's Task 4 requirement -- plain recall@k is reported
     for single-clause questions only.

This script does not touch the frozen eval JSON, does not change any
retrieval parameter, and does not query any LLM.

Run with: python -m scripts.recall_at_k_v2
"""

import json
from dataclasses import dataclass
from pathlib import Path

from src.config import RETRIEVAL
from scripts.ingest_large import CHROMA_DIR as LARGE_CHROMA_DIR
from scripts.ingest_large import COLLECTION_NAME as LARGE_COLLECTION_NAME
from scripts.recall_at_k_large import get_large_collection
from src.retrieval import hybrid, rerank, vector
from src.retrieval.vector import Chunk

V2_SET_PATH = Path("eval/frozen_retrieval_eval_v2_40.json")
ARMS = ("A", "B", "C")
ARM_LABELS = {"A": "dense", "B": "hybrid-RRF", "C": "hybrid+rerank"}
RECALL_KS = (1, 3, 5, 10)
POOL_K = RETRIEVAL.candidate_k  # 20, unchanged from primary/secondary experiments


def _retrieve(arm: str, question: str, collection) -> list[Chunk]:
    if arm == "A":
        return vector.retrieve(question, k=POOL_K, collection=collection)
    if arm == "B":
        return hybrid.retrieve(
            question, k=POOL_K, candidate_k=RETRIEVAL.candidate_k, rrf_k=RETRIEVAL.rrf_k, collection=collection
        )
    if arm == "C":
        candidate_pool = hybrid.retrieve(
            question, k=POOL_K, candidate_k=RETRIEVAL.candidate_k, rrf_k=RETRIEVAL.rrf_k, collection=collection
        )
        return rerank.rerank(question, candidate_pool, k=POOL_K)
    raise ValueError(f"unknown arm {arm!r}")


def _clause_tuple(clause: str) -> tuple[int, ...]:
    return tuple(int(p) for p in clause.split("."))


def _stored_label_covers(gold: str, stored_label: str) -> bool:
    if "-" in gold:
        return gold == stored_label
    if "-" not in stored_label:
        return gold == stored_label
    start, end = stored_label.split("-", 1)
    gold_t, start_t, end_t = _clause_tuple(gold), _clause_tuple(start), _clause_tuple(end)
    return start_t <= gold_t <= end_t


def _first_hit_rank(ranked_chunks: list[Chunk], gold_clause: str) -> int | None:
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if _stored_label_covers(gold_clause, chunk.clause):
            return rank
    return None


@dataclass
class QuestionResult:
    question_id: str
    arm: str
    gold_clauses: list[str]
    per_clause_first_hit_rank: dict[str, int | None]  # clause -> rank or None

    @property
    def is_multi_clause(self) -> bool:
        return len(self.gold_clauses) > 1

    @property
    def overall_first_hit_rank(self) -> int | None:
        """Best (lowest) rank across gold clauses -- used for single-clause
        Recall@K/MRR exactly like the original script; for multi-clause
        questions this is NOT sufficient on its own (see complete/partial/miss).
        """
        ranks = [r for r in self.per_clause_first_hit_rank.values() if r is not None]
        return min(ranks) if ranks else None

    def evidence_status(self, k: int) -> str:
        """complete / partial / miss at a given pool depth k -- multi-clause only."""
        hits = [r is not None and r <= k for r in self.per_clause_first_hit_rank.values()]
        if all(hits):
            return "complete"
        if any(hits):
            return "partial"
        return "miss"


def _load_questions() -> list[dict]:
    return json.loads(V2_SET_PATH.read_text())


def _extract_clause_only(gold: str) -> str:
    """Question 'source_clauses' entries are always bare clause numbers or
    literal merged-range labels (e.g. '2.10.8-2.10.9') -- never a '/'-joined
    chapter string (that only appears in the 'chapter' display field).
    """
    return gold


def run_diagnostic(collection, questions: list[dict]) -> list[QuestionResult]:
    results = []
    for q in questions:
        gold_clauses = [_extract_clause_only(c) for c in q["source_clauses"]]
        for arm in ARMS:
            ranked = _retrieve(arm, q["question"], collection)
            per_clause = {gold: _first_hit_rank(ranked, gold) for gold in gold_clauses}
            results.append(
                QuestionResult(
                    question_id=q["id"], arm=arm, gold_clauses=gold_clauses, per_clause_first_hit_rank=per_clause
                )
            )
    return results


def summarize_single_clause(results: list[QuestionResult], n_single: int) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_results = [r for r in results if r.arm == arm and not r.is_multi_clause]
        assert len(arm_results) == n_single, (arm, len(arm_results), n_single)
        row: dict[str, float] = {}
        for k in RECALL_KS:
            hits = sum(1 for r in arm_results if r.overall_first_hit_rank is not None and r.overall_first_hit_rank <= k)
            row[f"recall@{k}"] = hits / n_single
        mrr = sum((1.0 / r.overall_first_hit_rank) if r.overall_first_hit_rank else 0.0 for r in arm_results)
        row["mrr"] = mrr / n_single
        summary[arm] = row
    return summary


def summarize_multi_clause(results: list[QuestionResult], n_multi: int) -> dict[str, dict[str, dict[str, float]]]:
    """Per-arm, per-k complete/partial/miss RATES over the multi-clause subset."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for arm in ARMS:
        arm_results = [r for r in results if r.arm == arm and r.is_multi_clause]
        assert len(arm_results) == n_multi, (arm, len(arm_results), n_multi)
        per_k = {}
        for k in RECALL_KS:
            statuses = [r.evidence_status(k) for r in arm_results]
            per_k[f"k={k}"] = {
                "complete": statuses.count("complete") / n_multi,
                "partial": statuses.count("partial") / n_multi,
                "miss": statuses.count("miss") / n_multi,
            }
        summary[arm] = per_k
    return summary


def main() -> None:
    questions = _load_questions()
    n_total = len(questions)
    n_multi = sum(1 for q in questions if len(q["source_clauses"]) > 1)
    n_single = n_total - n_multi

    collection = get_large_collection()
    print(f"Corpus     : {LARGE_COLLECTION_NAME} @ {LARGE_CHROMA_DIR.resolve()} ({collection.count()} chunks)")
    print(f"Questions  : {n_total} ({V2_SET_PATH}, frozen) -- {n_single} single-clause, {n_multi} multi-clause")
    print(f"Pool depth : top-{POOL_K} per arm/question (RETRIEVAL.candidate_k)")
    print()

    results = run_diagnostic(collection, questions)

    single_summary = summarize_single_clause(results, n_single)
    print("=== Single-clause questions: Recall@K / MRR ===")
    metric_cols = [f"recall@{k}" for k in RECALL_KS] + ["mrr"]
    header = f"{'arm':<16}" + "".join(f"{m:>10}" for m in metric_cols)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        row = single_summary[arm]
        print(f"{arm+' ('+ARM_LABELS[arm]+')':<16}" + "".join(f"{row[m]:>10.3f}" for m in metric_cols))
    print()

    multi_summary = summarize_multi_clause(results, n_multi)
    print("=== Multi-clause questions: complete / partial / miss rate at each pool depth ===")
    for arm in ARMS:
        print(f"-- Arm {arm} ({ARM_LABELS[arm]}) --")
        for k in RECALL_KS:
            row = multi_summary[arm][f"k={k}"]
            print(f"  k={k:<3} complete={row['complete']:.3f}  partial={row['partial']:.3f}  miss={row['miss']:.3f}")
    print()

    # Write machine-readable output for downstream reporting.
    out = {
        "corpus": {"name": LARGE_COLLECTION_NAME, "chunk_count": collection.count()},
        "n_total": n_total,
        "n_single_clause": n_single,
        "n_multi_clause": n_multi,
        "single_clause_summary": single_summary,
        "multi_clause_summary": multi_summary,
        "per_question": [
            {
                "question_id": r.question_id,
                "arm": r.arm,
                "gold_clauses": r.gold_clauses,
                "per_clause_first_hit_rank": r.per_clause_first_hit_rank,
                "is_multi_clause": r.is_multi_clause,
            }
            for r in results
        ],
    }
    out_path = Path("reports/recall_v2_large_corpus.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Full per-question detail written to {out_path}")


if __name__ == "__main__":
    main()
