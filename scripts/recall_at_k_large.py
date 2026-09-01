"""SECONDARY EXPERIMENT -- local recall@k / MRR diagnostic, small corpus vs.
large corpus, for all three ablation arms. Zero API calls: retrieval and
reranking only (dense embeddings + cross-encoder reranking run locally).

Compares the primary 124-chunk `fssai_regulations` collection (data/chroma/)
against the new ~4x larger `fssai_large` collection (data/chroma_large/,
built by scripts/ingest_large.py) on the SAME 15 frozen questions and
clause-level ground truth from data/qa_set.json (read-only). Every
retrieval parameter (k_first_stage=20, k_final=3, rrf_k=60) is taken
unchanged from src.config.RETRIEVAL -- the only variable across the two
columns of the printed table is which collection is queried.

Recall@k here is computed over a ranked pool of RETRIEVAL.candidate_k (20)
chunks per arm/question -- deep enough to score recall@10 -- then scored at
prefix lengths 1/3/5/10. This does not change production behavior: arm A's
actual generation call still requests only k_final chunks (see
eval/run_generation.py); this script separately asks each retriever for a
deeper ranked list purely to measure how far down the ranking the gold
clause sits.

A chunk "hits" a question if its stored clause label covers any of that
question's gold source_clauses -- exact match, or containment within a
merged-range label (e.g. gold "2.4.4" is covered by stored label
"2.4.4-2.4.5"), the same short-clause-merge format src.chunking produces.

Run with: python -m scripts.recall_at_k_large
"""

import json
from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import RETRIEVAL
from src.ingest import CHROMA_DIR as PRIMARY_CHROMA_DIR
from src.ingest import COLLECTION_NAME as PRIMARY_COLLECTION_NAME
from src.ingest import EMBEDDING_MODEL_NAME, get_collection
from src.retrieval import hybrid, rerank, vector
from src.retrieval.vector import Chunk
from scripts.ingest_large import CHROMA_DIR as LARGE_CHROMA_DIR
from scripts.ingest_large import COLLECTION_NAME as LARGE_COLLECTION_NAME

QA_SET_PATH = Path("data/qa_set.json")
ARMS = ("A", "B", "C")
ARM_LABELS = {
    "A": "dense",
    "B": "hybrid-RRF",
    "C": "hybrid+rerank",
}
RECALL_KS = (1, 3, 5, 10)
POOL_K = RETRIEVAL.candidate_k  # 20: deep enough for recall@10, unchanged from the primary experiment


def get_large_collection() -> Collection:
    """Read-only handle on the secondary experiment's `fssai_large`
    collection -- separate from scripts.ingest_large.get_large_collection()
    only in that it never upserts.
    """
    client = chromadb.PersistentClient(path=str(LARGE_CHROMA_DIR))
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
    return client.get_collection(name=LARGE_COLLECTION_NAME, embedding_function=embedding_function)


def _retrieve(arm: str, question: str, collection: Collection) -> list[Chunk]:
    """Same three arm definitions as eval/run_generation.py, but each asked
    for a POOL_K-deep ranked list instead of just k_final, against the
    given `collection`.
    """
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
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def _clause_tuple(clause: str) -> tuple[int, ...]:
    return tuple(int(p) for p in clause.split("."))


def _stored_label_covers(gold: str, stored_label: str) -> bool:
    """True if `gold` (e.g. "2.4.4") is covered by `stored_label`, which is
    either an exact clause number or a merged-range label produced by
    src.chunking's short-clause merging (e.g. "2.4.4-2.4.5"). Mirrors
    scripts/ingest_large.py's identical helper.
    """
    if "-" not in stored_label:
        return gold == stored_label
    start, end = stored_label.split("-", 1)
    gold_t, start_t, end_t = _clause_tuple(gold), _clause_tuple(start), _clause_tuple(end)
    return start_t <= gold_t <= end_t


def _first_hit_rank(ranked_chunks: list[Chunk], gold_clauses: list[str]) -> int | None:
    """1-indexed rank of the first chunk covering any gold clause, or None
    if no chunk in the ranked pool covers one.
    """
    for rank, chunk in enumerate(ranked_chunks, start=1):
        if any(_stored_label_covers(gold, chunk.clause) for gold in gold_clauses):
            return rank
    return None


@dataclass
class QuestionResult:
    question_id: str
    arm: str
    first_hit_rank: int | None


def _load_questions() -> list[dict]:
    return json.loads(QA_SET_PATH.read_text())


def run_diagnostic(collection: Collection, questions: list[dict]) -> list[QuestionResult]:
    results = []
    for q in questions:
        for arm in ARMS:
            ranked = _retrieve(arm, q["question"], collection)
            results.append(
                QuestionResult(question_id=q["id"], arm=arm, first_hit_rank=_first_hit_rank(ranked, q["source_clauses"]))
            )
    return results


def summarize(results: list[QuestionResult], n_questions: int) -> dict[str, dict[str, float]]:
    """Per-arm recall@k (for each k in RECALL_KS) and MRR, averaged over
    n_questions.
    """
    summary: dict[str, dict[str, float]] = {}
    for arm in ARMS:
        arm_results = [r for r in results if r.arm == arm]
        assert len(arm_results) == n_questions
        row: dict[str, float] = {}
        for k in RECALL_KS:
            hits = sum(1 for r in arm_results if r.first_hit_rank is not None and r.first_hit_rank <= k)
            row[f"recall@{k}"] = hits / n_questions
        mrr = sum((1.0 / r.first_hit_rank) if r.first_hit_rank is not None else 0.0 for r in arm_results)
        row["mrr"] = mrr / n_questions
        summary[arm] = row
    return summary


def print_comparison_table(small: dict[str, dict[str, float]], large: dict[str, dict[str, float]]) -> None:
    metric_cols = [f"recall@{k}" for k in RECALL_KS] + ["mrr"]
    header = f"{'arm':<16}{'corpus':<8}" + "".join(f"{m:>10}" for m in metric_cols)
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        label = f"{arm} ({ARM_LABELS[arm]})"
        small_row = small[arm]
        large_row = large[arm]
        print(f"{label:<16}{'small':<8}" + "".join(f"{small_row[m]:>10.3f}" for m in metric_cols))
        print(f"{'':<16}{'large':<8}" + "".join(f"{large_row[m]:>10.3f}" for m in metric_cols))
        deltas = [large_row[m] - small_row[m] for m in metric_cols]
        print(f"{'':<16}{'delta':<8}" + "".join(f"{d:>+10.3f}" for d in deltas))


def main() -> None:
    questions = _load_questions()
    n = len(questions)

    primary_collection = get_collection()
    large_collection = get_large_collection()

    print(f"Small corpus : {PRIMARY_COLLECTION_NAME} @ {PRIMARY_CHROMA_DIR.resolve()} ({primary_collection.count()} chunks)")
    print(f"Large corpus : {LARGE_COLLECTION_NAME} @ {LARGE_CHROMA_DIR.resolve()} ({large_collection.count()} chunks)")
    print(f"Questions    : {n} (data/qa_set.json, frozen)")
    print(f"Pool depth   : top-{POOL_K} per arm/question (RETRIEVAL.candidate_k)")
    print()

    small_results = run_diagnostic(primary_collection, questions)
    large_results = run_diagnostic(large_collection, questions)

    small_summary = summarize(small_results, n)
    large_summary = summarize(large_results, n)

    print_comparison_table(small_summary, large_summary)


if __name__ == "__main__":
    main()
