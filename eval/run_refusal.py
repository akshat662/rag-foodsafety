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
"""

import argparse
import json
import time
from pathlib import Path

from eval.run_generation import ARMS, MAX_GENERATION_RETRIES, _generate_with_retries, _git_commit, _retrieve

REFUSAL_SET_PATH = Path("data/refusal_set.json")
RUNS_DIR = Path("runs")


def load_refusal_questions() -> list[dict]:
    return json.loads(REFUSAL_SET_PATH.read_text())


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


def _run_one(question: dict, arm: str) -> dict:
    question_id = question["id"]
    question_text = question["question"]

    t0 = time.monotonic()
    try:
        chunks = _retrieve(arm, question_text)
    except Exception as exc:  # noqa: BLE001
        return {"question_id": question_id, "arm": arm, "status": "error", "stage": "retrieval", "error": repr(exc)}
    t1 = time.monotonic()

    try:
        result, attempts = _generate_with_retries(question_text, chunks)
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


def run(run_id: str, arms: tuple[str, ...]) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "refusal_results.jsonl"

    (run_dir / "config.json").write_text(
        json.dumps({"arms": list(arms), "source": str(REFUSAL_SET_PATH)}, indent=2)
    )
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")

    questions = load_refusal_questions()
    completed = _load_completed(output_path)

    total_pairs = len(questions) * len(arms)
    done = skipped = 0

    with output_path.open("a") as out:
        for question in questions:
            for arm in arms:
                if (question["id"], arm) in completed:
                    skipped += 1
                    continue
                row = _run_one(question, arm)
                out.write(json.dumps(row) + "\n")
                out.flush()
                done += 1
                print(f"[{done + skipped}/{total_pairs}] {question['id']} arm={arm} status={row['status']}")

    print(f"Done. {done} generated, {skipped} skipped. Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None, help="Run ID under runs/. Default: new timestamp.")
    parser.add_argument("--arms", default=",".join(ARMS), help="Comma-separated arms to test, e.g. A,B,C.")
    args = parser.parse_args()

    run_id = args.run_id or f"refusal_{time.strftime('%Y%m%d_%H%M%S')}"
    run(run_id, tuple(args.arms.split(",")))


if __name__ == "__main__":
    main()
