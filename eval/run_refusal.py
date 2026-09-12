"""Day 3 Step 8: refusal / abstention test on out-of-scope questions.

Reads data/refusal_set.json (frozen, read-only) -- 5 questions from FSSAI
domains that were never ingested into this project's corpus (honey, edible
oils, canned fish, fruit jam, carbonated water) -- and runs each one through
the existing retrieval + generation pipeline, under each of the three
ablation arms (reusing eval.run_generation's arm definitions and retry
logic, not duplicating them). Checks whether the system abstains rather
than answering from the generator's own parametric memory, which is what
data/refusal_set.json's notes already showed it will confidently do with no
retrieval in front of it.

No RAGAS, no OpenAI: these questions have no ground truth to score against
(the corpus that would contain the real answer isn't ingested), only an
expected_behavior of "abstain" -- scoring faithfulness/relevancy/precision/
recall against a nonexistent reference would be meaningless. Per the Day 3
plan (section 9): do not put the refusal set through RAGAS.

This demonstrates verified abstention behaviour on out-of-scope questions --
it is not a general hallucination benchmark, and should not be reported as
one.

Run with: python -m eval.run_refusal [--run-id RUN_ID] [--arms A,B,C]
    [--refusal-set PATH] [--collection-name NAME] [--chroma-dir DIR]

The --refusal-set/--collection-name/--chroma-dir overrides (added for the v2
evaluation pass) change nothing about retrieval or generation logic -- they
only let this script point at a different frozen question file and a
different (already-built) Chroma collection, mirroring the equivalent
overrides eval/run_generation.py already exposed for the secondary corpus-
scale experiment.
"""

import argparse
import json
import time
from pathlib import Path

import chromadb
from chromadb import Collection
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from eval.run_generation import (
    ARMS,
    GENERATOR_CHOICES,
    MAX_GENERATION_RETRIES,
    _generate_with_retries,
    _git_commit,
    _retrieve,
)
from src.config import GENERATOR, ProviderConfig
from src.ingest import COLLECTION_NAME as PRIMARY_COLLECTION_NAME
from src.ingest import EMBEDDING_MODEL_NAME

REFUSAL_SET_PATH = Path("data/refusal_set.json")
RUNS_DIR = Path("runs")


def load_refusal_questions(path: Path = REFUSAL_SET_PATH) -> list[dict]:
    return json.loads(path.read_text())


def _load_completed(output_path: Path) -> set[tuple[str, str]]:
    completed: set[tuple[str, str]] = set()
    if not output_path.exists():
        return completed
    for line in output_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            completed.add((row["question_id"], row["arm"]))
    return completed


def _run_one(
    question: dict,
    arm: str,
    collection: Collection | None = None,
    generator_config: ProviderConfig = GENERATOR,
) -> dict:
    question_id = question["id"]
    question_text = question["question"]

    t0 = time.monotonic()
    try:
        chunks = _retrieve(arm, question_text, collection=collection)
    except Exception as exc:  # noqa: BLE001
        return {"question_id": question_id, "arm": arm, "status": "error", "stage": "retrieval", "error": repr(exc)}
    t1 = time.monotonic()

    try:
        result, attempts = _generate_with_retries(question_text, chunks, generator_config=generator_config)
    except Exception as exc:  # noqa: BLE001
        return {
            "question_id": question_id,
            "arm": arm,
            "status": "error",
            "stage": "generation",
            "error": repr(exc),
            "retrieved_clauses": [c.clause for c in chunks],
        }
    t2 = time.monotonic()

    expected = question["expected_behavior"]
    refused_correctly = result.abstained and expected == "abstain"

    return {
        "question_id": question_id,
        "arm": arm,
        "status": "ok",
        "expected_behavior": expected,
        "abstained": result.abstained,
        "refused_correctly": refused_correctly,
        "retrieved_clauses": [c.clause for c in chunks],
        "answer": result.answer,
        "citations": result.citations,
        "generation_attempts": attempts,
        "latency_ms": {
            "retrieval": round((t1 - t0) * 1000, 1),
            "generation": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        },
    }


def run(
    run_id: str,
    arms: tuple[str, ...],
    refusal_set_path: Path = REFUSAL_SET_PATH,
    collection: Collection | None = None,
    generator_config: ProviderConfig = GENERATOR,
) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "refusal_results.jsonl"

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "arms": list(arms),
                "source": str(refusal_set_path),
                "generator": {"provider": generator_config.provider, "model": generator_config.model},
                "collection": {
                    "name": collection.name if collection is not None else PRIMARY_COLLECTION_NAME,
                    "chunk_count": collection.count() if collection is not None else None,
                },
            },
            indent=2,
        )
    )
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")

    questions = load_refusal_questions(refusal_set_path)
    completed = _load_completed(output_path)

    total_pairs = len(questions) * len(arms)
    done = skipped = 0

    with output_path.open("a") as out:
        for question in questions:
            for arm in arms:
                if (question["id"], arm) in completed:
                    skipped += 1
                    continue
                row = _run_one(question, arm, collection=collection, generator_config=generator_config)
                out.write(json.dumps(row) + "\n")
                out.flush()
                done += 1
                print(f"[{done + skipped}/{total_pairs}] {question['id']} arm={arm} status={row['status']}")

    print(f"Done. {done} generated, {skipped} skipped. Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Run ID under runs/. Default: new timestamp.")
    parser.add_argument("--arms", default=",".join(ARMS), help="Comma-separated arms to test, e.g. A,B,C.")
    parser.add_argument(
        "--refusal-set",
        default=str(REFUSAL_SET_PATH),
        help="Path to the frozen refusal question set (default: data/refusal_set.json).",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Override the Chroma collection name (default: fssai_regulations, primary).",
    )
    parser.add_argument(
        "--chroma-dir",
        default=None,
        help="Override the Chroma persist directory (default: data/chroma, primary).",
    )
    parser.add_argument(
        "--generator",
        default="groq",
        choices=list(GENERATOR_CHOICES),
        help="Which generator config to use (default: groq, the frozen Day 3 Qwen generator).",
    )
    args = parser.parse_args()

    collection: Collection | None = None
    if args.collection_name or args.chroma_dir:
        client = chromadb.PersistentClient(path=args.chroma_dir or "data/chroma")
        embedding_function = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL_NAME)
        collection = client.get_collection(
            name=args.collection_name or PRIMARY_COLLECTION_NAME,
            embedding_function=embedding_function,
        )

    run_id = args.run_id or f"refusal_{time.strftime('%Y%m%d_%H%M%S')}"
    run(
        run_id,
        tuple(args.arms.split(",")),
        refusal_set_path=Path(args.refusal_set),
        collection=collection,
        generator_config=GENERATOR_CHOICES[args.generator],
    )


if __name__ == "__main__":
    main()
