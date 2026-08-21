"""Day 3 Stage 2: RAGAS judging over the frozen Stage 1 generations.

Reads runs/<stage1_run_id>/generations.jsonl (read-only) and data/qa_set.json
(read-only), scores every successful (question, arm) pair on four RAGAS
metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall)
using an OpenAI judge model, and writes one JSON line per
(question_id, arm, metric) to runs/<ragas_run_id>/scores.jsonl. Makes zero
retrieval or generation calls -- Stage 1 is frozen input here, never
modified. Never reads data/refusal_set.json: refusal questions have no
ground truth to judge against.

Two modes:

  --dry-run   Estimates LLM call count, token usage, and cost from the
              ACTUAL 45 saved Stage 1 records using RAGAS's real prompt
              templates (via each metric's PydanticPrompt.to_string), then
              tiktoken. Makes no network call of any kind -- no OpenAI
              client is even constructed in this path. This is the only
              mode Day 3 Stage 2 infrastructure work has been asked to run.

  (default)   The real judging run: incremental, resumable, budget-guarded.
              Requires OPENAI_API_KEY. Not invoked until explicitly
              authorized.

Run with: python -m eval.run_ragas --dry-run [--stage1-run RUN_ID] [--pilot]
"""

import argparse
import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import tiktoken

import ragas
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from ragas.metrics._context_precision import QAC
from ragas.metrics._context_recall import QCA
from ragas.metrics._faithfulness import NLIStatementInput, StatementGeneratorInput
from ragas.metrics._answer_relevance import ResponseRelevanceInput

QA_SET_PATH = Path("data/qa_set.json")
RUNS_DIR = Path("runs")

# ---------------------------------------------------------------------------
# Judge model candidates and pricing.
#
# Fetched live from developers.openai.com/api/docs/pricing on 2026-08-20 via
# WebFetch (my training data predates today by ~7 months, so this was
# verified rather than recalled). RE-VERIFY AT platform.openai.com/pricing
# BEFORE purchasing credits -- OpenAI pricing has changed multiple times in
# 2026 and this file's numbers are a snapshot, not a live source.
# Prices are USD per 1,000,000 tokens, standard (non-batch) tier.
# ---------------------------------------------------------------------------
JUDGE_MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-5-nano": {"input": 0.05, "output": 0.40},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
}

# Selected judge (2026-08-21, supersedes the 2026-08-20 gpt-4o-mini
# recommendation -- see decisions.md for both writeups):
#   gpt-5-mini, reasoning_effort="minimal".
# The cost differential vs. gpt-4o-mini is insignificant against the
# $2-3 project budget (both are well under $1 for the full run -- see the
# per-model comparison this module can still produce for any candidate in
# JUDGE_MODEL_PRICING_PER_1M). The deciding factor was model
# lifecycle/reproducibility: gpt-4o-mini is an older-generation model
# family; gpt-5-mini is current-generation, avoiding dependence on a
# family that may enter wind-down before this project is done being
# referenced. reasoning_effort=minimal because the judge task is
# structured NLI-style evaluation (decompose into statements, verify each
# against context, short JSON verdict) rather than open-ended reasoning --
# the lowest non-"none" effort tier is the appropriate match.
# Pending pilot validation, not yet treated as fully frozen the way
# GENERATOR is -- src.config.JUDGE.model is set, but actual reasoning-token
# usage under "minimal" is unverified until the pilot runs (see the
# REASONING_TOKENS_PER_CALL_ASSUMPTION caveat below).
PROPOSED_JUDGE_MODEL = "gpt-5-mini"
PROPOSED_JUDGE_REASONING_EFFORT = "minimal"

# Verified (not assumed) via tiktoken.encoding_for_model(), which has an
# explicit entry for "gpt-5-mini" in the installed tiktoken==0.14.0's model
# registry (not a fallback/guess): it maps to o200k_base, the same encoding
# used for gpt-4o-mini and every other 2026-era OpenAI chat model checked.
# Unchanged from the prior (gpt-4o-mini) estimate -- switching judge models
# did not require regenerating input-token counts.
TOKEN_ENCODING_NAME = "o200k_base"

# UPDATED 2026-08-21 from the 6-sample pilot's MEASURED usage: all 24
# successful (question, arm, metric) judge calls -- Faithfulness, Answer
# Relevancy, Context Precision, Context Recall -- reported
# usage_metadata.output_token_details.reasoning == 0 at
# reasoning_effort="minimal", with zero exceptions (see
# runs/ragas_pilot_20260821_101134/scores.jsonl). The original value of
# 100 (see decisions.md, 2026-08-20) was a pre-pilot planning assumption,
# not a measurement, and it materially overstated cost: pilot actual cost
# ($0.0234) came in ~37% under the pre-call estimate that used it
# ($0.0370), almost entirely because of this one number. Set to 0 now that
# real evidence exists. Caveat: the pilot covered only 2 of 15 questions
# (q01, q02 -- both single-clause sibling_disambiguation/lexical
# questions); harder multi-hop questions (q13-q15, needing two source
# clauses) were not exercised and could plausibly push some reasoning
# tokens above zero during the full 45-pair run. The real run should keep
# recording actual reasoning tokens per call (already wired via
# _UsageCapture) rather than trusting this constant blindly.
REASONING_TOKENS_PER_CALL_ASSUMPTION = 0

# RAGAS's own default (3) trades a little output-token cost for averaging
# over 3 reverse-engineered questions per answer. Set to 1 here -- NOT
# primarily for cost (see the dry-run report: strictness only changes an
# `n=` parameter on a single API call, so the $ delta between 1 and 3 is
# negligible at these prices) but for simplicity and consistency with the
# rest of the eval treating n=15/arm as already noise-limited (Day 3 plan
# section 8: "differences below ~0.03 at n=15 are noise").
ANSWER_RELEVANCY_STRICTNESS = 1

# Local embedding model for Answer Relevancy's cosine-similarity step
# (BAAI/bge-small-en-v1.5, same model src/ingest.py uses) -- never OpenAI's
# embeddings API, so Answer Relevancy contributes zero embedding cost.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

BUDGET_CAP_USD = 3.0
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# Stage 1 / QA set loading -- read-only.
# ---------------------------------------------------------------------------


def load_stage1_rows(stage1_run_id: str) -> list[dict]:
    """Load successful Stage 1 rows only. Never writes to the Stage 1 run dir."""
    path = RUNS_DIR / stage1_run_id / "generations.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("status") == "ok"]


def load_qa_lookup() -> dict[str, dict]:
    """question_id -> {question, ground_truth} from the frozen QA set. Read-only."""
    questions = json.loads(QA_SET_PATH.read_text())
    return {q["id"]: q for q in questions}


# ---------------------------------------------------------------------------
# Metric objects (construction only -- no llm/embeddings wired in, so this
# is safe to do in dry-run mode with no API key and no network access).
# ---------------------------------------------------------------------------


def build_metrics() -> dict[str, Any]:
    return {
        "faithfulness": Faithfulness(),
        "answer_relevancy": AnswerRelevancy(strictness=ANSWER_RELEVANCY_STRICTNESS),
        "context_precision": ContextPrecision(),
        "context_recall": ContextRecall(),
    }


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _approximate_statements(answer: str) -> list[str]:
    """Stand-in for Faithfulness's statement-generation output, used ONLY to
    size the second (NLI verification) call's input for the dry-run cost
    estimate -- we don't have real generated statements without calling the
    LLM. Naive sentence splitting is a reasonable proxy here because Stage 1
    answers are short, single- or double-sentence factual statements (see
    runs/stage1_.../generations.jsonl), which is exactly the shape
    Faithfulness's statement generator tends to preserve 1:1 for inputs this
    simple. Flagged as an approximation, not measured ground truth.
    """
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(answer.strip()) if s.strip()]
    return sentences or [answer.strip()]


# ---------------------------------------------------------------------------
# Output-token estimation.
#
# Input tokens below are EXACT: they tokenize the real prompt text RAGAS's
# PydanticPrompt.to_string() renders (instruction + JSON schema + few-shot
# examples + our actual Stage 1 data). Output tokens are NOT measured --
# no completion has been generated -- and are estimated from the shape of
# each prompt's real few-shot example outputs (printed via the installed
# ragas 0.4.3 metric objects) and the sentence count of the actual input.
# These are the least certain numbers in this report and should be treated
# as a planning estimate to be corrected by the pilot's real usage, not a
# guarantee.
# ---------------------------------------------------------------------------

# Tokens per statement/sentence in a verdict-style JSON output (verdict +
# 1-sentence reason), calibrated against the printed NLIStatementOutput /
# ContextRecallClassifications / Verification few-shot examples.
_OUTPUT_TOKENS_PER_VERDICT = 35
_OUTPUT_TOKENS_STATEMENT_GEN_BASE = 10
_OUTPUT_TOKENS_PER_GENERATED_STATEMENT = 15
_OUTPUT_TOKENS_ANSWER_RELEVANCY = 25  # short reverse-engineered question + noncommittal flag


def _encoding():
    return tiktoken.get_encoding(TOKEN_ENCODING_NAME)


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def _is_reasoning_model(model: str) -> bool:
    """GPT-5-family models bill hidden reasoning tokens as output tokens,
    even at reasoning_effort="minimal" (see REASONING_TOKENS_PER_CALL_ASSUMPTION).
    Non-reasoning models (gpt-4o-mini, gpt-4.1-*) do not.
    """
    return model.startswith("gpt-5")


@dataclass
class MetricCallEstimate:
    calls: int
    input_tokens: int
    output_tokens: int  # visible JSON output only
    reasoning_tokens: int = 0  # hidden, billed-as-output; 0 for non-reasoning judge models

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.reasoning_tokens


def estimate_pair(
    row: dict, qa: dict, metrics: dict[str, Any], enc, judge_model: str = PROPOSED_JUDGE_MODEL
) -> dict[str, MetricCallEstimate]:
    """Per-metric call/token estimate for one (question, arm) Stage 1 row."""
    question = qa[row["question_id"]]["question"]
    ground_truth = qa[row["question_id"]]["ground_truth"]
    answer = row["answer"]
    contexts = row["contexts"]

    reasoning_per_call = REASONING_TOKENS_PER_CALL_ASSUMPTION if _is_reasoning_model(judge_model) else 0

    out: dict[str, MetricCallEstimate] = {}

    # --- Faithfulness: 2 calls (statement generation, then NLI verification) ---
    f = metrics["faithfulness"]
    stmt_prompt_text = f.statement_generator_prompt.to_string(
        StatementGeneratorInput(question=question, answer=answer)
    )
    statements = _approximate_statements(answer)
    nli_prompt_text = f.nli_statements_prompt.to_string(
        NLIStatementInput(context="\n".join(contexts), statements=statements)
    )
    stmt_in = count_tokens(stmt_prompt_text, enc)
    nli_in = count_tokens(nli_prompt_text, enc)
    stmt_out = _OUTPUT_TOKENS_STATEMENT_GEN_BASE + _OUTPUT_TOKENS_PER_GENERATED_STATEMENT * len(statements)
    nli_out = _OUTPUT_TOKENS_PER_VERDICT * len(statements)
    out["faithfulness"] = MetricCallEstimate(
        calls=2,
        input_tokens=stmt_in + nli_in,
        output_tokens=stmt_out + nli_out,
        reasoning_tokens=reasoning_per_call * 2,
    )

    # --- Answer Relevancy: 1 call (n=strictness choices in one request) ---
    ar = metrics["answer_relevancy"]
    ar_prompt_text = ar.question_generation.to_string(ResponseRelevanceInput(response=answer))
    ar_in = count_tokens(ar_prompt_text, enc)
    ar_out = _OUTPUT_TOKENS_ANSWER_RELEVANCY * ANSWER_RELEVANCY_STRICTNESS
    out["answer_relevancy"] = MetricCallEstimate(
        calls=1, input_tokens=ar_in, output_tokens=ar_out, reasoning_tokens=reasoning_per_call
    )

    # --- Context Precision: 1 call PER retrieved chunk ---
    cp = metrics["context_precision"]
    cp_in = 0
    for chunk in contexts:
        text = cp.context_precision_prompt.to_string(QAC(question=question, context=chunk, answer=ground_truth))
        cp_in += count_tokens(text, enc)
    cp_out = _OUTPUT_TOKENS_PER_VERDICT * len(contexts)
    out["context_precision"] = MetricCallEstimate(
        calls=len(contexts),
        input_tokens=cp_in,
        output_tokens=cp_out,
        reasoning_tokens=reasoning_per_call * len(contexts),
    )

    # --- Context Recall: 1 call over the full concatenated context ---
    cr = metrics["context_recall"]
    cr_prompt_text = cr.context_recall_prompt.to_string(
        QCA(question=question, context="\n".join(contexts), answer=ground_truth)
    )
    cr_in = count_tokens(cr_prompt_text, enc)
    gt_sentences = _approximate_statements(ground_truth)
    cr_out = _OUTPUT_TOKENS_PER_VERDICT * len(gt_sentences)
    out["context_recall"] = MetricCallEstimate(
        calls=1, input_tokens=cr_in, output_tokens=cr_out, reasoning_tokens=reasoning_per_call
    )

    return out


def _cost_usd(input_tokens: int, output_tokens: int, reasoning_tokens: int, pricing: dict[str, float]) -> float:
    # Reasoning tokens are billed at the output-token rate (OpenAI: hidden
    # reasoning tokens occupy context and are billed as output tokens).
    return (input_tokens / 1_000_000) * pricing["input"] + (
        (output_tokens + reasoning_tokens) / 1_000_000
    ) * pricing["output"]


def dry_run_estimate(
    stage1_run_id: str,
    judge_model: str,
    pilot_question_ids: set[str] | None = None,
) -> dict:
    """Full cost estimate over the 45 (or pilot-subset) Stage 1 rows.

    Makes no network call and constructs no OpenAI client -- pricing is a
    static local table (JUDGE_MODEL_PRICING_PER_1M), and prompt rendering
    uses only the already-instantiated ragas Metric objects' local prompt
    templates plus tiktoken.
    """
    if judge_model not in JUDGE_MODEL_PRICING_PER_1M:
        raise ValueError(f"unknown judge model {judge_model!r}; add pricing to JUDGE_MODEL_PRICING_PER_1M")
    pricing = JUDGE_MODEL_PRICING_PER_1M[judge_model]
    is_reasoning = _is_reasoning_model(judge_model)

    rows = load_stage1_rows(stage1_run_id)
    qa = load_qa_lookup()
    metrics = build_metrics()
    enc = _encoding()

    per_metric_totals: dict[str, dict[str, int]] = {
        name: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0} for name in metrics
    }
    per_pair_costs: list[dict] = []
    pair_estimate_cache: dict[tuple[str, str], dict[str, MetricCallEstimate]] = {}

    for row in rows:
        estimates = estimate_pair(row, qa, metrics, enc, judge_model=judge_model)
        pair_estimate_cache[(row["question_id"], row["arm"])] = estimates
        pair_cost = 0.0
        for name, est in estimates.items():
            per_metric_totals[name]["calls"] += est.calls
            per_metric_totals[name]["input_tokens"] += est.input_tokens
            per_metric_totals[name]["output_tokens"] += est.output_tokens
            per_metric_totals[name]["reasoning_tokens"] += est.reasoning_tokens
            pair_cost += _cost_usd(est.input_tokens, est.output_tokens, est.reasoning_tokens, pricing)
        per_pair_costs.append(
            {"question_id": row["question_id"], "arm": row["arm"], "estimated_cost_usd": round(pair_cost, 6)}
        )

    per_metric_report = {}
    total_cost = 0.0
    for name, totals in per_metric_totals.items():
        cost = _cost_usd(totals["input_tokens"], totals["output_tokens"], totals["reasoning_tokens"], pricing)
        total_cost += cost
        per_metric_report[name] = {
            "calls": totals["calls"],
            "input_tokens": totals["input_tokens"],
            "output_tokens_visible": totals["output_tokens"],
            "output_tokens_reasoning_assumed": totals["reasoning_tokens"],
            "total_tokens": totals["input_tokens"] + totals["output_tokens"] + totals["reasoning_tokens"],
            "estimated_cost_usd": round(cost, 4),
        }

    n_pairs = len(rows)

    # Pilot subset: either an explicit set of question_ids, or (default) the
    # first 2 distinct question_ids in Stage 1 order x all 3 arms = 6 pairs,
    # matching the Stage 1 smoke-test convention (q01, q06).
    if pilot_question_ids is None:
        seen: list[str] = []
        for r in rows:
            if r["question_id"] not in seen:
                seen.append(r["question_id"])
            if len(seen) == 2:
                break
        pilot_question_ids = set(seen)
    pilot_rows = [r for r in rows if r["question_id"] in pilot_question_ids]
    pilot_cost = sum(
        _cost_usd(
            sum(e.input_tokens for e in pair_estimate_cache[(r["question_id"], r["arm"])].values()),
            sum(e.output_tokens for e in pair_estimate_cache[(r["question_id"], r["arm"])].values()),
            sum(e.reasoning_tokens for e in pair_estimate_cache[(r["question_id"], r["arm"])].values()),
            pricing,
        )
        for r in pilot_rows
    )

    # Worst case: every call retries once (MAX_RETRIES's practical ceiling
    # for planning is "everything needs exactly one extra attempt", not
    # MAX_RETRIES-1 retries on every single call -- that would be a
    # near-total-failure scenario, not a "worst reasonable case"). This is
    # a conservative multiplier on top of an already-conservative per-call
    # estimate, not a hard ceiling RAGAS is guaranteed to respect -- see
    # caveats below.
    worst_case_cost = total_cost * 2

    report = {
        "ragas_version": ragas.__version__,
        "judge_model": judge_model,
        "judge_reasoning_effort": PROPOSED_JUDGE_REASONING_EFFORT if is_reasoning else None,
        "is_reasoning_model": is_reasoning,
        "pricing_per_1m_tokens_usd": pricing,
        "pricing_source": (
            "developers.openai.com/api/docs/pricing, fetched 2026-08-20 -- "
            "RE-VERIFY before purchasing credits"
        ),
        "token_encoding_assumption": TOKEN_ENCODING_NAME,
        "token_encoding_verified_via": "tiktoken.encoding_for_model() explicit registry entry, not a fallback",
        "reasoning_tokens_per_call_assumption": REASONING_TOKENS_PER_CALL_ASSUMPTION if is_reasoning else 0,
        "stage1_run_id": stage1_run_id,
        "n_pairs_total": n_pairs,
        "n_pairs_pilot": len(pilot_rows),
        "pilot_question_ids": sorted(pilot_question_ids),
        "answer_relevancy_strictness": ANSWER_RELEVANCY_STRICTNESS,
        "context_precision_k": len(rows[0]["contexts"]) if rows else None,
        "per_metric": per_metric_report,
        "total_expected_llm_calls": sum(v["calls"] for v in per_metric_report.values()),
        "total_estimated_input_tokens": sum(v["input_tokens"] for v in per_metric_report.values()),
        "total_estimated_output_tokens_visible": sum(
            v["output_tokens_visible"] for v in per_metric_report.values()
        ),
        "total_estimated_output_tokens_reasoning_assumed": sum(
            v["output_tokens_reasoning_assumed"] for v in per_metric_report.values()
        ),
        "total_estimated_cost_usd_full_45": round(total_cost, 4),
        "estimated_cost_usd_pilot_6": round(pilot_cost, 4),
        "worst_case_cost_usd_full_45": round(worst_case_cost, 4),
        "budget_cap_usd": BUDGET_CAP_USD,
        "within_budget_full_run": total_cost <= BUDGET_CAP_USD,
        "within_budget_worst_case": worst_case_cost <= BUDGET_CAP_USD,
        "caveats": [
            "Input tokens are exact (tiktoken over RAGAS's real rendered prompts using actual Stage 1 data).",
            "Visible-output tokens are ESTIMATED from few-shot example shapes, not measured -- true values "
            "will differ; the pilot run's actual usage should replace these numbers before trusting the "
            "full-45 projection.",
            "Reasoning tokens (gpt-5-mini only) are an UNVERIFIED planning assumption "
            f"({REASONING_TOKENS_PER_CALL_ASSUMPTION} tokens/call at reasoning_effort=minimal), not measured "
            "or documented by OpenAI as a specific number -- OpenAI confirms reasoning tokens are billed as "
            "output tokens and that minimal effort still reasons adaptively, but gives no concrete token "
            "count. The pilot's real usage.completion_tokens_details.reasoning_tokens must replace this "
            "before the full-45 projection is trusted.",
            "generate_multiple's retry behavior (retries_left, default 3 inside RAGAS itself, separate "
            "from this harness's own MAX_RETRIES) is not reflected per-call above; worst_case applies a "
            "flat 2x multiplier as a coarse upper bound, not a simulation of RAGAS's internal retry logic.",
            "Token encoding verified via tiktoken.encoding_for_model(judge_model) rather than assumed; "
            "still, exact tokenizer behavior for very new models can occasionally lag tiktoken releases.",
        ],
    }
    return report


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write_dry_run_artifacts(run_dir: Path, report: dict, stage1_run_id: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "dry_run_estimate.json").write_text(json.dumps(report, indent=2))

    stage1_git_commit = (RUNS_DIR / stage1_run_id / "git_commit.txt").read_text().strip()
    config = {
        "ragas_version": report["ragas_version"],
        "judge_model": report["judge_model"],
        "judge_reasoning_effort": report["judge_reasoning_effort"],
        "judge_model_status": (
            "SELECTED, pending pilot validation -- src.config.JUDGE.model is set to this value; "
            "see decisions.md 2026-08-21 for the gpt-4o-mini -> gpt-5-mini decision"
        ),
        "metrics": {
            "faithfulness": "ragas.metrics.Faithfulness(), defaults",
            "answer_relevancy": f"ragas.metrics.AnswerRelevancy(strictness={ANSWER_RELEVANCY_STRICTNESS})",
            "context_precision": "ragas.metrics.ContextPrecision() -- reference-based (requires ground_truth)",
            "context_recall": "ragas.metrics.ContextRecall() -- reference-based (requires ground_truth)",
        },
        "embedding_model_for_answer_relevancy": EMBEDDING_MODEL_NAME,
        "pricing_source": report["pricing_source"],
        "budget_cap_usd": BUDGET_CAP_USD,
        "source_stage1_run_id": stage1_run_id,
        "source_stage1_git_commit": stage1_git_commit,
        "git_commit": _git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")


# ---------------------------------------------------------------------------
# Real-run scaffolding: budget guard + incremental/resumable scoring.
#
# Structurally complete per the Stage 2 infrastructure requirements, but NOT
# invoked by --dry-run and not invoked by this module's CLI unless a mode
# other than --dry-run is explicitly requested. Requires OPENAI_API_KEY.
# ---------------------------------------------------------------------------


@dataclass
class BudgetGuard:
    """Tracks cumulative estimated spend against BUDGET_CAP_USD and refuses
    to authorize further judge calls once the projected total would exceed
    it. Never silently continues past the cap: callers must check
    `can_afford` before every call and stop the run (preserving whatever was
    already written) the first time it returns False.
    """

    cap_usd: float
    pricing: dict[str, float]
    spent_usd: float = field(default=0.0)
    stop_reason: str | None = field(default=None)

    def can_afford(
        self, estimated_input_tokens: int, estimated_output_tokens: int, estimated_reasoning_tokens: int = 0
    ) -> bool:
        projected = self.spent_usd + _cost_usd(
            estimated_input_tokens, estimated_output_tokens, estimated_reasoning_tokens, self.pricing
        )
        if projected > self.cap_usd:
            self.stop_reason = (
                f"projected spend ${projected:.4f} would exceed budget cap ${self.cap_usd:.2f} "
                f"(already spent ${self.spent_usd:.4f})"
            )
            return False
        return True

    def record(self, input_tokens: int, output_tokens: int, reasoning_tokens: int = 0) -> None:
        self.spent_usd += _cost_usd(input_tokens, output_tokens, reasoning_tokens, self.pricing)


def _load_completed_scores(output_path: Path) -> set[tuple[str, str, str]]:
    """(question_id, arm, metric) triples already scored with status='ok'."""
    completed: set[tuple[str, str, str]] = set()
    if not output_path.exists():
        return completed
    for line in output_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            completed.add((row["question_id"], row["arm"], row["metric"]))
    return completed


class _UsageCapture:
    """LangChain callback handler capturing the real token usage of the most
    recent LLM call made through it. Passed per-call via ragas's own
    `callbacks=` parameter on single_turn_score/ascore, so usage is
    attributed to exactly the (question, arm, metric) call that used it,
    not blended across a whole run.
    """

    def __init__(self):
        self.usage: dict | None = None

    def __call__(self):
        from langchain_core.callbacks import BaseCallbackHandler

        outer = self

        class Handler(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):
                try:
                    gen = response.generations[0][0]
                    usage = getattr(getattr(gen, "message", None), "usage_metadata", None)
                    if usage:
                        outer.usage = dict(usage)
                except Exception:  # noqa: BLE001
                    pass

        return Handler()


def _actual_tokens_from_usage(usage: dict | None) -> tuple[int, int, int]:
    """(input_tokens, visible_output_tokens, reasoning_tokens) from a
    LangChain usage_metadata dict. output_tokens there is the TOTAL
    completion tokens billed (visible content + hidden reasoning combined),
    matching OpenAI's own accounting -- see decisions.md, 2026-08-21 pilot
    entry, for the calibration call that established this shape.
    """
    if not usage:
        return 0, 0, 0
    input_tokens = usage.get("input_tokens", 0)
    output_total = usage.get("output_tokens", 0)
    reasoning = usage.get("output_token_details", {}).get("reasoning", 0)
    return input_tokens, max(0, output_total - reasoning), reasoning


def run_real_evaluation(
    stage1_run_id: str,
    ragas_run_id: str,
    judge_model: str,
    reasoning_effort: str | None = None,
    question_ids: set[str] | None = None,
    max_retries: int = MAX_RETRIES,
    budget_cap_usd: float = BUDGET_CAP_USD,
) -> dict:
    """The actual paid judging run.

    Requires OPENAI_API_KEY. Imports openai/langchain_openai lazily, inside
    this function only, so `python -m eval.run_ragas --dry-run` never
    imports them and can run with zero API key and zero network access.
    `question_ids`, when given, restricts to those Stage 1 rows only (the
    pilot subset) -- all three arms for each included question_id still run.
    """
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. This is the real (paid) RAGAS judging run -- "
            "refusing to proceed without an explicit key."
        )

    # NOT ragas.embeddings.HuggingFaceEmbeddings (capital F): that's the
    # newer BaseRagasEmbedding class (embed_text() only). The legacy
    # AnswerRelevancy metric requires the older BaseRagasEmbeddings
    # interface (embed_query()/embed_documents()) -- confirmed live during
    # this pilot's second attempt, which failed every answer_relevancy call
    # with AttributeError: 'HuggingFaceEmbeddings' object has no attribute
    # 'embed_query'. LangchainEmbeddingsWrapper adapts any langchain
    # Embeddings object to that older interface; langchain_community's
    # HuggingFaceEmbeddings runs bge-small locally via sentence-transformers,
    # same as everywhere else in this project -- no OpenAI embedding calls.
    from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from langchain_openai import ChatOpenAI

    pricing = JUDGE_MODEL_PRICING_PER_1M[judge_model]
    guard = BudgetGuard(cap_usd=budget_cap_usd, pricing=pricing)

    run_dir = RUNS_DIR / ragas_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "scores.jsonl"

    stage1_git_commit = (RUNS_DIR / stage1_run_id / "git_commit.txt").read_text().strip()
    run_config = {
        "run_id": ragas_run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": _git_commit(),
        "ragas_version": ragas.__version__,
        "judge_model": judge_model,
        "judge_reasoning_effort": reasoning_effort or (PROPOSED_JUDGE_REASONING_EFFORT if _is_reasoning_model(judge_model) else None),
        "metrics": {
            "faithfulness": "ragas.metrics.Faithfulness(), defaults",
            "answer_relevancy": f"ragas.metrics.AnswerRelevancy(strictness={ANSWER_RELEVANCY_STRICTNESS})",
            "context_precision": "ragas.metrics.ContextPrecision() -- reference-based",
            "context_recall": "ragas.metrics.ContextRecall() -- reference-based",
        },
        "embedding_model_for_answer_relevancy": EMBEDDING_MODEL_NAME,
        "source_stage1_run_id": stage1_run_id,
        "source_stage1_git_commit": stage1_git_commit,
        "pilot_question_ids": sorted(question_ids) if question_ids else None,
        "budget_cap_usd": budget_cap_usd,
    }
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    (run_dir / "git_commit.txt").write_text(_git_commit() + "\n")

    # Reasoning models (gpt-5-family) take reasoning_effort, not temperature.
    # gpt-5-mini goes further: it REJECTS any temperature value other than
    # its default (1) with a 400 (confirmed live during this pilot's first,
    # failed attempt -- see decisions.md, 2026-08-21 pilot entry). RAGAS's
    # own LangchainLLMWrapper.generate() has a hardcoded temperature=0.01
    # default it applies to every call regardless of how the ChatOpenAI
    # object was constructed, so omitting temperature from chat_kwargs
    # alone is not enough -- bypass_temperature=True is ragas's documented
    # escape hatch (LangchainLLMWrapper.__init__) for exactly this case:
    # it skips ragas's temperature override entirely, leaving the
    # judge model to use its own default.
    if _is_reasoning_model(judge_model):
        chat_kwargs = {"model": judge_model, "reasoning_effort": reasoning_effort or "minimal"}
    else:
        chat_kwargs = {"model": judge_model, "temperature": 0}
    llm = LangchainLLMWrapper(ChatOpenAI(**chat_kwargs), bypass_temperature=_is_reasoning_model(judge_model))
    embeddings = LangchainEmbeddingsWrapper(LCHuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME))

    metrics = build_metrics()
    for metric in metrics.values():
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings

    rows = load_stage1_rows(stage1_run_id)
    if question_ids is not None:
        rows = [r for r in rows if r["question_id"] in question_ids]
    qa = load_qa_lookup()
    enc = _encoding()
    completed = _load_completed_scores(output_path)

    n_attempted = 0
    n_skipped = 0
    n_api_calls = 0
    stopped_early = False
    with output_path.open("a") as out:
        for row in rows:
            # estimate_pair always computes all 4 metrics' estimates in one
            # pass (it doesn't accept a subset) -- compute once per row, not
            # once per metric, and index into the result below.
            row_estimates = estimate_pair(row, qa, metrics, enc, judge_model=judge_model)
            for metric_name, metric in metrics.items():
                key = (row["question_id"], row["arm"], metric_name)
                if key in completed:
                    n_skipped += 1
                    continue

                est = row_estimates[metric_name]
                if not guard.can_afford(est.input_tokens, est.output_tokens, est.reasoning_tokens):
                    stopped_early = True
                    break

                result = _score_one_metric(row, qa, metric_name, metric, max_retries, pricing, est)
                n_attempted += 1
                n_api_calls += result.get("attempts", 1)
                # Only record real spend for a call that actually succeeded
                # and returned billable usage. A failed call in this harness
                # means every attempt errored (typically a request-level
                # rejection before OpenAI processes any tokens, as with the
                # gpt-5-mini temperature 400 hit during this pilot's first
                # attempt) -- charging the pre-call ESTIMATE against the
                # budget for a request that was never billed would silently
                # inflate spent_usd with money that was never spent.
                if result["status"] == "ok":
                    tokens = result.get("tokens", {})
                    guard.record(
                        tokens.get("input", est.input_tokens),
                        tokens.get("output_visible", est.output_tokens),
                        tokens.get("reasoning", est.reasoning_tokens),
                    )
                out.write(json.dumps(result) + "\n")
                out.flush()
            if stopped_early:
                break

    if stopped_early:
        print(f"STOPPED: {guard.stop_reason}")
        print(f"Partial results preserved in {output_path}; re-run with the same --run-id to resume.")

    return {
        "run_dir": str(run_dir),
        "n_attempted": n_attempted,
        "n_skipped": n_skipped,
        "n_api_calls": n_api_calls,
        "spent_usd": round(guard.spent_usd, 6),
        "stopped_early": stopped_early,
        "stop_reason": guard.stop_reason,
    }


def _score_one_metric(
    row: dict,
    qa: dict,
    metric_name: str,
    metric: Any,
    max_retries: int,
    pricing: dict[str, float],
    est: "MetricCallEstimate",
) -> dict:
    """One (question, arm, metric) judge call with retry-on-failure, real
    token-usage capture (via a per-call LangChain callback), and both
    estimated (dry-run) and actual cost recorded side by side.
    """
    from ragas import SingleTurnSample

    question = qa[row["question_id"]]["question"]
    sample = SingleTurnSample(
        user_input=question,
        retrieved_contexts=row["contexts"],
        response=row["answer"],
        reference=qa[row["question_id"]]["ground_truth"],
    )

    t0 = time.monotonic()
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        capture = _UsageCapture()
        handler = capture()
        try:
            score = metric.single_turn_score(sample, callbacks=[handler])
            t1 = time.monotonic()
            in_tok, out_vis_tok, reason_tok = _actual_tokens_from_usage(capture.usage)
            actual_cost = _cost_usd(in_tok, out_vis_tok, reason_tok, pricing) if capture.usage else None
            estimated_cost = _cost_usd(est.input_tokens, est.output_tokens, est.reasoning_tokens, pricing)
            return {
                "question_id": row["question_id"],
                "arm": row["arm"],
                "metric": metric_name,
                "status": "ok",
                "score": score,
                "attempts": attempt,
                "latency_ms": round((t1 - t0) * 1000, 1),
                "tokens": {
                    "input": in_tok,
                    "output_visible": out_vis_tok,
                    "reasoning": reason_tok,
                    "source": "actual" if capture.usage else "estimate-fallback",
                },
                "estimated_cost_usd": round(estimated_cost, 6),
                "actual_cost_usd": round(actual_cost, 6) if actual_cost is not None else None,
            }
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    return {
        "question_id": row["question_id"],
        "arm": row["arm"],
        "metric": metric_name,
        "status": "error",
        "error": repr(last_exc),
        "attempts": max_retries,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-run", required=True, help="Source Stage 1 run_id under runs/.")
    parser.add_argument("--run-id", default=None, help="Ragas run id under runs/. Default: new timestamp.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate cost only. No API calls, no key needed.")
    parser.add_argument(
        "--real", action="store_true", help="Execute the actual (paid) judging run. Requires OPENAI_API_KEY."
    )
    parser.add_argument("--judge-model", default=PROPOSED_JUDGE_MODEL, choices=list(JUDGE_MODEL_PRICING_PER_1M))
    parser.add_argument("--reasoning-effort", default=PROPOSED_JUDGE_REASONING_EFFORT)
    parser.add_argument("--pilot", action="store_true", help="Restrict --dry-run to the 6-pair pilot subset only.")
    parser.add_argument(
        "--pilot-questions",
        default=None,
        help="Comma-separated question_ids to restrict --real to (e.g. q01,q02). All 3 arms still run for each.",
    )
    parser.add_argument("--budget-cap", type=float, default=BUDGET_CAP_USD)
    args = parser.parse_args()

    if args.dry_run and args.real:
        raise SystemExit("--dry-run and --real are mutually exclusive.")
    if not args.dry_run and not args.real:
        raise SystemExit("Specify exactly one of --dry-run or --real.")

    run_id = args.run_id or f"ragas_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_id

    if args.dry_run:
        report = dry_run_estimate(args.stage1_run, args.judge_model)
        write_dry_run_artifacts(run_dir, report, args.stage1_run)
        print(json.dumps(report, indent=2))
        print(f"\nWritten to {run_dir}/")
        return

    question_ids = set(args.pilot_questions.split(",")) if args.pilot_questions else None
    summary = run_real_evaluation(
        stage1_run_id=args.stage1_run,
        ragas_run_id=run_id,
        judge_model=args.judge_model,
        reasoning_effort=args.reasoning_effort,
        question_ids=question_ids,
        budget_cap_usd=args.budget_cap,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
