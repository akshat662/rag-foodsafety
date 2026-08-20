"""Day 3 Stage 1: per-arm, per-question grounded generation.

For each of the 15 frozen questions in data/qa_set.json, retrieves context
under each of the three ablation arms and generates a grounded answer via
src.generate (which itself routes through src.llm / the rate limiter). Writes
one JSON line per (question, arm) to runs/<run_id>/generations.jsonl
immediately on completion -- this file is both Stage 1's output and its
resume cache.

Arms (src.config.RETRIEVAL is the single source of truth for k_final /
candidate_k / rrf_k, shared by all three):
    A - dense only, top k_final
    B - dense + BM25 -> RRF, top k_final
    C - dense + BM25 -> RRF candidate pool (candidate_k) -> cross-encoder
        rerank -> top k_final

Makes no RAGAS / judging calls -- that is a separate, paid stage
(eval/run_ragas.py) that reads this file's output.

Run with: python -m eval.run_generation [--run-id RUN_ID] [--questions PATH]
"""

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.config import GENERATOR, RETRIEVAL
from src.generate import generate
from src.ingest import EMBEDDING_MODEL_NAME
from src.rate_limiter import estimate_tokens
from src.retrieval import hybrid, rerank, vector
from src.retrieval.vector import Chunk

ARMS = ("A", "B", "C")
MAX_GENERATION_RETRIES = 3
QA_SET_PATH = Path("data/qa_set.json")
RUNS_DIR = Path("runs")


def _retrieve(arm: str, question: str) -> list[Chunk]:
    """Retrieve context chunks for `question` under `arm`, per the Day 3 plan's
    fixed arm definitions. Every arm shares src.config.RETRIEVAL's k_final, so
    only the retrieval/ranking method differs between arms, never context size.
    """
    if arm == "A":
        return vector.retrieve(question, k=RETRIEVAL.k_final)
    if arm == "B":
        return hybrid.retrieve(
            question,
            k=RETRIEVAL.k_final,
            candidate_k=RETRIEVAL.candidate_k,
            rrf_k=RETRIEVAL.rrf_k,
        )
    if arm == "C":
        candidate_pool = hybrid.retrieve(
            question,
            k=RETRIEVAL.candidate_k,
            candidate_k=RETRIEVAL.candidate_k,
            rrf_k=RETRIEVAL.rrf_k,
        )
        return rerank.rerank(question, candidate_pool, k=RETRIEVAL.k_final)
    raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")


def _chunk_id(chunk: Chunk) -> str:
    """Stable identifier for a retrieved chunk.

    src.retrieval.vector.Chunk carries no ingestion-time chunk ID (that ID
    only exists inside src.ingest, keyed on a part_index the retrieval layer
    doesn't expose), so this derives one from the same fields
    src.retrieval.hybrid uses to dedupe (chapter, clause, page, text) --
    identical chunks always produce the identical ID. A sha1 digest is used
    instead of Python's built-in hash() because str hashing is randomized
    per-process (PYTHONHASHSEED); a resumed run must reproduce the same IDs.
    """
    key = f"{chunk.chapter}:{chunk.clause}:{chunk.page}:{chunk.text}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def _generate_with_retries(question: str, chunks: list[Chunk]) -> tuple[Any, int]:
    """Call src.generate.generate, retrying up to MAX_GENERATION_RETRIES times.

    The rate limiter (src/rate_limiter.py) already retries 429s internally;
    it does not retry malformed structured output (e.g. the model returning
    non-JSON), which is a distinct failure mode observed during Day 2's
    smoke test (see DECISIONS.md, 2026-08-19). That is retried here instead.
    """
    last_exc: Exception | None = None
    for attempt in range(1, MAX_GENERATION_RETRIES + 1):
        try:
            return generate(question, chunks), attempt
        except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure retries
            last_exc = exc
    raise last_exc


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_questions(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def _load_completed(output_path: Path) -> set[tuple[str, str]]:
    """(question_id, arm) pairs already written with status "ok".

    Error rows are deliberately excluded so a resumed run retries pairs that
    failed last time, rather than leaving them permanently unretried.
    """
    completed: set[tuple[str, str]] = set()
    if not output_path.exists():
        return completed
    with output_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") == "ok":
                completed.add((row["question_id"], row["arm"]))
    return completed


def _write_run_metadata(run_dir: Path) -> None:
    config = {
        "generator": {
            "provider": GENERATOR.provider,
            "model": GENERATOR.model,
            "temperature": GENERATOR.temperature,
        },
        "retrieval": asdict(RETRIEVAL),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "arms": {
            "A": "dense only, top k_final",
            "B": "dense + BM25 -> RRF, top k_final",
            "C": "dense + BM25 -> RRF candidate pool (candidate_k) -> cross-encoder rerank -> top k_final",
        },
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")


def _run_one(question: dict, arm: str) -> dict:
    question_id = question["id"]
    question_text = question["question"]

    t0 = time.monotonic()
    try:
        chunks = _retrieve(arm, question_text)
    except Exception as exc:  # noqa: BLE001 - one bad pair must not kill the run
        return {
            "question_id": question_id,
            "arm": arm,
            "status": "error",
            "stage": "retrieval",
            "error": repr(exc),
        }
    t1 = time.monotonic()

    try:
        result, attempts = _generate_with_retries(question_text, chunks)
    except Exception as exc:  # noqa: BLE001 - one bad pair must not kill the run
        return {
            "question_id": question_id,
            "arm": arm,
            "status": "error",
            "stage": "generation",
            "error": repr(exc),
            "retrieved_clauses": [c.clause for c in chunks],
            "retrieved_chunk_ids": [_chunk_id(c) for c in chunks],
        }
    t2 = time.monotonic()

    prompt_tokens = estimate_tokens(question_text) + sum(estimate_tokens(c.text) for c in chunks)
    output_tokens = estimate_tokens(result.answer)

    return {
        "question_id": question_id,
        "arm": arm,
        "status": "ok",
        "retrieved_clauses": [c.clause for c in chunks],
        "retrieved_chunk_ids": [_chunk_id(c) for c in chunks],
        "contexts": [c.text for c in chunks],
        "answer": result.answer,
        "citations": result.citations,
        "abstained": result.abstained,
        "generation_attempts": attempts,
        "latency_ms": {
            "retrieval": round((t1 - t0) * 1000, 1),
            "generation": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        },
        "tokens": {
            # Estimated (src.rate_limiter.estimate_tokens), not provider-reported:
            # src/llm.py's uniform interface returns parsed content only, not
            # per-call usage, and CLAUDE.md forbids calling the provider SDK
            # directly from here to recover it.
            "prompt_estimated": prompt_tokens,
            "output_estimated": output_tokens,
        },
    }


def run(run_id: str, questions_path: Path) -> None:
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "generations.jsonl"

    _write_run_metadata(run_dir)

    questions = _load_questions(questions_path)
    completed = _load_completed(output_path)

    total_pairs = len(questions) * len(ARMS)
    skipped = 0
    done = 0

    with output_path.open("a") as out:
        for question in questions:
            for arm in ARMS:
                if (question["id"], arm) in completed:
                    skipped += 1
                    continue

                row = _run_one(question, arm)
                out.write(json.dumps(row) + "\n")
                out.flush()
                done += 1
                print(
                    f"[{done + skipped}/{total_pairs}] "
                    f"{question['id']} arm={arm} status={row['status']}"
                )

    print(f"Done. {done} generated, {skipped} skipped (already complete). Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run ID under runs/. Reuse an existing one to resume. Default: new timestamp.",
    )
    parser.add_argument(
        "--questions",
        default=str(QA_SET_PATH),
        help="Path to the frozen QA set (default: data/qa_set.json).",
    )
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    run(run_id, Path(args.questions))


if __name__ == "__main__":
    main()
