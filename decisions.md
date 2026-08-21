# Project Decisions

## 2026-08-18 — API Rate Limits

### Gemini 3.6 Flash — RAG Generator

* Provider: Google Gemini API
* Model: `Gemini 3.6 Flash`
* RPM: **5 requests/minute**
* TPM: **250,000 tokens/minute**
* RPD: **20 requests/day**
* Source: Google AI Studio → Rate Limit dashboard
* Usage tier: Free Tier
* Recorded: 2026-08-18

### Groq `openai/gpt-oss-120b` — RAGAS Judge

* Provider: Groq
* Model: `openai/gpt-oss-120b`
* RPM: **30 requests/minute**
* TPM: **8,000 tokens/minute**
* RPD: **1,000 requests/day**
* Tokens/day: **200,000 tokens/day**
* Release stage: Production
* Source: Groq model limits
* Recorded: 2026-08-18

### Decision

Evaluation runs will be scheduled according to the live rate limits recorded above rather than published or third-party estimates. The Gemini generator is constrained primarily by its 5 RPM and 20 RPD limits, while the Groq RAGAS judge has substantially higher request capacity but only 8K TPM. These project/account-specific limits will be treated as the baseline for planning evaluation batches and preventing rate-limit failures.

## 2026-08-18 — Corpus source

FSSAI Compendium on FSS (Food Products Standards and Food Additives)
Regulation, 2011 — Chapters 2.1 (Dairy), 2.4 (Cereals), 2.9 (Salt/Spices).

Source: fssai.gov.in, compendium page, last updated 15/05/2025.

Downloaded 18 Aug 2026. Version pinned; amendments after this date not tracked.

Chosen for parallel clause structure across product categories (reranker
stressor) and numeric-limit density (BM25 stressor).

## 2026-08-18 — Extraction quality

pymupdf extraction: clause text and numeric values interpretable.
Table formatting artifacts present — column alignment degrades in wide
tables. Consequence: avoid QA pairs whose answers depend on reading a
specific cell from a mangled wide table. Noted as a README limitation.

## 2026-08-18 — Memorization check: 5/5, corpus RETAINED (decision revised)
Gemini 3.6 Flash answered 5/5 headline facts with no context, grounding off.
Initially treated as a fail. Revised after considering which metrics are
affected:
- Context precision / context recall are retrieval-only — uncontaminated.
  The ablation tests retrieval, so the headline result stands.
- Faithfulness/answer-relevancy are affected; noted as a limitation and
  reported on a non-memorized subset as well.
- Test facts were the most republished figures in Indian food regulation.
  QA set will be written from obscure clauses instead, with a
  parametric_known flag per question measured at authoring time.
- 5/5 memorization strengthens the abstention test: the model provably
  knows these facts, so correct refusal proves grounding is enforced.

  ## 2026-08-19 — Generation robustness observation

A transient Groq 400/malformed-generation response occurred during the
5-question smoke test. A retry succeeded on the next attempt.

The current rate limiter retries rate-limit errors but does not retry
malformed structured-output responses. This is not being changed at the
generation layer yet; the future evaluation harness must handle an
individual failed generation without terminating the complete ablation run.

## 2026-08-19 — Refusal set frozen (5 questions)

8 candidate refusal questions were drafted from FSSAI domains not present in
the ingested corpus (honey, edible oils, meat, fish, fruit/veg products,
beverages, confectionery). Each was sent to the actual RAG generator
(Groq gpt-oss-20b) with no retrieved context, no RAG pipeline, and no
grounding prompt, to measure parametric knowledge.

All 8 candidates produced confident, specific, non-hedging answers with no
uncertainty language. Rather than accept all 8 at face value, each answer
was checked for plausibility against general food-standards knowledge
(Codex Alimentarius / internationally-consistent figures), since the actual
FSSAI chapters for these domains are outside our corpus and cannot be
verified directly. One candidate (honey minimum reducing sugar, "18%") was
judged implausible against known honey composition norms (~60-65%
internationally) and excluded as a likely hallucination despite its
confident phrasing. Two more (chicken sausage moisture, confectionery SO2)
were excluded for citing regulation section numbers that could not be
corroborated and looked potentially fabricated.

Final 5 selected: honey moisture (20%), sunflower oil FFA (0.4%), canned
fish histamine (200 mg/kg), fruit jam minimum Brix (~66°), carbonated water
caffeine (150 mg/L) — all marked `parametric_known: true` because the
generator answered confidently with no context, not because the figures
have been verified as FSSAI-correct. `data/refusal_set.json` explicitly
records each raw model answer and notes that the underlying number is
unverified against the real regulation text.

No retrieval, reranking, generation-with-context, or RAGAS was run against
these questions before freezing.

## 2026-08-20 — Deterministic generation for Day 3 Stage 1

`src.config.ProviderConfig` gained a `temperature: float | None = None` field.
`GENERATOR` (groq / `openai/gpt-oss-20b`) now sets `temperature=0.0`.
`GroqLLMClient` (src/llm.py) passes `temperature=...` through to both
`generate` and `generate_structured` Groq calls whenever the active
`ProviderConfig.temperature` is not `None` (a `None` config, e.g. the
not-yet-implemented `JUDGE`, keeps the provider's own default untouched).

Added because `eval/run_generation.py` (Day 3 Stage 1) requires
reproducible runs: re-running generation for the same (question, arm) pair
must yield the same answer, or run artifacts and their git-hash provenance
stop being comparable across re-runs. `GEN_PROVIDER`/`GEN_MODEL` are
unchanged (groq / `openai/gpt-oss-20b`); the OpenAI-judge / same-lab
question (Day 3 plan Step 4) is still open and unaffected by this change.

## 2026-08-20 — Reportable generator model: Qwen

`src.config.GENERATOR.model` changed from `openai/gpt-oss-20b` (used for
the first Stage 1 smoke test) to `qwen/qwen3.6-27b`, both on Groq. Provider
stays `groq`; `temperature` stays `0.0`; no other config, retrieval
parameter, prompt, or frozen data file changed.

Reason: the RAGAS judge (`src.config.JUDGE`) will be an OpenAI model.
Keeping `gpt-oss-20b` (OpenAI's open-weight family) as the generator would
put the generator and judge in the same model family, reviving a
same-lab/self-preference-bias criticism of the eval (Day 3 plan, Step 4).
Qwen gives clean separation: generator (Alibaba/Qwen) and judge (OpenAI)
are different labs.

`qwen/qwen3.6-27b` is currently a Groq **preview** model — subject to
change/deprecation without the stability guarantee of a GA model. Verified
live against the project's Groq account before switching: present in
`GET /openai/v1/models`, and a completion call succeeded with
`x-ratelimit-limit-requests: 1000` / `x-ratelimit-limit-tokens: 8000`,
matching the RPD/TPM already configured in `RateLimits` for `GENERATOR` —
no rate-limit values were changed. Free-plan daily budget used for
planning: **200K TPD**, same allowance as `gpt-oss-20b` (per Groq account
limits, not independently re-verified via a header — Groq's rate-limit
headers exposed request/minute and token/minute caps but not a daily-token
header).

Observation for the Day 3 run (not a blocking issue): `qwen/qwen3.6-27b` is
a reasoning model that emits a `<think>...</think>` block in
`message.content` on a plain free-text completion. The pipeline is
unaffected because `src.generate.generate` always calls
`llm.generate_structured` (`response_format: json_object`), and under
JSON mode the verified response content was clean, directly-parseable
JSON with no reasoning leakage. Reasoning does consume hidden completion
tokens the rate limiter's pre-call token estimate (prompt-only) doesn't
account for, which may cause more real 429 backoff during the 45-pair run
than under `gpt-oss-20b` — expected to be absorbed by the existing
retry/backoff, not a correctness risk.

## 2026-08-21 — Stage 2 (RAGAS) infrastructure: version, packaging fix, judge candidates

`ragas==0.4.3` installed (latest on PyPI at install time), per Day 3 plan
section 4. Confirmed installed metric class names before writing any
harness code: `ragas.metrics.Faithfulness`, `AnswerRelevancy`,
`ContextPrecision`, `ContextRecall` all exist and match the legacy
class-based API the rest of the RAGAS ecosystem's tutorials assume.
`ContextPrecision`'s default instance requires `reference` (ground truth)
in `_required_columns` — confirmed by inspecting the installed class
directly, not assumed: it is the reference-based variant, consistent with
scoring against `qa_set.json`'s `ground_truth` field.

**Packaging issue hit and fixed**: `ragas==0.4.3` fails to import at all
(`ModuleNotFoundError` cascading to `ImportError`) against the
latest `langchain-community==0.4.2`, because `ragas/llms/base.py`
unconditionally imports `langchain_community.chat_models.vertexai.ChatVertexAI`
at module load time regardless of which provider is actually used, and
`langchain-community` removed that submodule entirely in its 0.4.x
"sunset" releases. Fixed by pinning `langchain-community==0.3.31` (the
last release that still ships it) — no code workaround, no ragas patch.
Logged here per the Day 3 plan's explicit warning that RAGAS's API/dependency
churn "is a reliable way to lose two hours" — this cost about fifteen
minutes because it was diagnosed by reading the actual traceback and
`ragas`'s import line rather than guessing.

**Exact per-metric LLM call counts**, confirmed by reading the installed
`_ascore`/`_ascore` implementations directly (not assumed from the Day 3
plan's planning table, though they turned out to match):
- Faithfulness: 2 calls (statement generation, then NLI verification).
- Answer Relevancy: **1 call**, not `strictness` calls — `strictness`
  is passed as `n=strictness` on a single `generate_multiple` request, so
  it only inflates output tokens (a handful of extra short JSON completions),
  not input tokens or call count. Set to `strictness=1` anyway, for
  defensibility/simplicity rather than cost (the dollar delta between
  strictness=1 and strictness=3 is negligible at these prices — see the
  dry-run report). This corrects an initial assumption that reducing
  strictness was primarily a cost lever; it is not, materially.
- Context Precision: 1 call per retrieved chunk (`k_final=3` → 3 calls).
- Context Recall: 1 call over the full concatenated context.
- Total: 7 calls/pair × 45 pairs = 315 calls, matching the Day 3 plan's
  original estimate exactly once Answer Relevancy's real call count (not
  its worst-case-per-strictness reading) is used.

**Judge model candidates evaluated** (pricing fetched live via WebFetch
from developers.openai.com/api/docs/pricing on 2026-08-20 — my training
data predates today by ~7 months and was not used for pricing numbers;
re-verify at platform.openai.com/pricing before purchasing credits):
`gpt-4o-mini`, `gpt-5-nano`, `gpt-5-mini`, `gpt-4.1-nano`, `gpt-4.1-mini`.
Full-45-pair dry-run cost estimates ranged $0.038 (gpt-5-nano) to $0.285
(gpt-4.1-mini) — all candidates are comfortably inside the $2–3 budget by
more than an order of magnitude, so cost was not the deciding factor
between them.

**Proposed** (not yet confirmed — `src.config.JUDGE.model` is still
`"TBD"`): `gpt-4o-mini`. Nano-tier models (`gpt-5-nano`, `gpt-4.1-nano`)
were set aside on capability grounds, not price: faithfulness/context-precision
judging is a multi-step NLI-style task (decompose into statements, verify
each against context, produce a reasoned verdict), and nano-tier models are
the least proven at this specific kind of structured judgment. `gpt-4o-mini`
is the most widely validated LLM-as-judge choice in the published
RAG-evaluation/RAGAS literature specifically, which matters for defensibility
in an interview/academic setting. Between the two viable mini-tier options,
`gpt-4o-mini`'s input price ($0.15/1M) beats `gpt-5-mini`'s ($0.25/1M) on
our actual, heavily input-token-dominated workload (long regulation-text
contexts, short JSON verdicts), so it is also the cheaper of the two
credible choices, not only the more established one.

**Dry-run cost estimate for the real 45-pair evaluation** (`gpt-4o-mini`,
against the frozen `stage1_20260820_215228` run): 315 calls, ~672K input /
~10K output tokens, **$0.107 estimated, $0.214 worst-case (2x margin)** —
full report in `runs/ragas_20260821_002531/dry_run_estimate.json`. No
OpenAI API call was made to produce this; input tokens are exact (tiktoken
over RAGAS's real rendered prompts), output tokens are a documented
estimate pending pilot-run correction.

## 2026-08-19 — Evaluation set frozen (15 questions)

15 eval questions written to `data/qa_set.json`: 4 lexical / 3 semantic /
5 sibling_disambiguation / 3 multi_hop, across chapters 2.1 (6), 2.4 (5),
2.9 (4). Authored entirely from `data/processed/*.txt` with no retrieval,
reranking, generation, or RAGAS run against any question during authoring.
Ground truth written in our own words with units; source_clauses use
regulatory clause numbers only, never chunk IDs.

Parametric-memory check run afterward against the actual generator (Groq
gpt-oss-20b), no context, no RAG: 3/15 questions matched the model's
unguided answer (q02 Table Butter milk fat 80%, q10 Whey Powder moisture
5%, q12 Cloves powder acid-insoluble ash 0.5% -- all cases where FSSAI's
figure coincides with a common, round, internationally-standard number).
The other 12/15 were not reproduced by the model, which instead answered
confidently with wrong numbers and, in several cases, fabricated citations
to standards that could not be corroborated (e.g. a fake "IS 1120:1969"
for Maida gluten, a fake "ISO 17043:2005" for mustard AITC content, a fake
"IS 15692:2014" for Atta/Maida ash). This validates the plan's authoring
strategy (obscure clause-level precision over "most republished figures")
directly: the eval set is 80% clean of parametric contamination for
faithfulness/answer-relevancy scoring, and the fabricated-citation pattern
is itself worth reporting as a finding about ungrounded generation.

Full review checklist passed: exact question/type/chapter counts, every
source_text and ground_truth filled, every source_clauses entry non-empty,
parametric_known filled for all 20 questions (15 eval + 5 refusal), both
JSON files parse cleanly, no duplicate IDs.

`data/qa_set.json` and `data/refusal_set.json` are committed together as a
single frozen snapshot, before any retrieval, reranking, generation-with-
context, or RAGAS evaluation runs against them.

## 2026-08-21 — Judge model: gpt-5-mini over gpt-4o-mini

Selected gpt-5-mini over gpt-4o-mini. The cost differential was
insignificant against the project budget. The deciding factor was model
lifecycle and reproducibility: using a current-generation mini-tier model
avoids dependence on a model family potentially entering wind-down.
reasoning_effort was set to minimal because the judge task is structured
NLI-style evaluation rather than open-ended reasoning. The pilot will
validate actual output-token usage including reasoning tokens.

`src.config.JUDGE` updated: `model="gpt-5-mini"`, `reasoning_effort="minimal"`.
Both `gpt-4o-mini` and `gpt-5-mini` full-45-pair dry-run estimates stay
under $0.30 (see below) — two orders of magnitude under the $2-3 cap either
way, which is why cost was not the deciding factor between them.

Two things checked before accepting this change, not assumed:

- **Tokenizer**: `tiktoken==0.14.0`'s `encoding_for_model("gpt-5-mini")`
  returns `o200k_base` — an explicit registry entry, not a silent
  fallback — identical to the encoding already used for `gpt-4o-mini`.
  Input-token counts did not need to be regenerated; they are exact either
  way (tiktoken over RAGAS's real rendered prompts).
- **Reasoning tokens**: unlike gpt-4o-mini, gpt-5-mini is a reasoning model.
  OpenAI's own docs (developers.openai.com/api/docs/guides/reasoning,
  fetched 2026-08-21) confirm reasoning tokens are billed as output tokens
  even though invisible in the response, and that `reasoning_effort=minimal`
  still reasons "adaptively" rather than skipping reasoning entirely — but
  give no concrete per-call token count. `eval/run_ragas.py` therefore
  carries an explicit, separately-reported
  `REASONING_TOKENS_PER_CALL_ASSUMPTION = 100` tokens/call, flagged in the
  dry-run report's `caveats` as unverified and loosely anchored against
  Stage 1's own live observation of a different reasoning model (Qwen3.6 on
  Groq generating ~200 reasoning tokens for a trivial, undialed request),
  scaled down for `minimal` effort and our short, templated per-call tasks.
  This is exactly the number the pilot is expected to correct.

**Updated dry-run cost estimate** (gpt-5-mini, reasoning_effort=minimal,
against the frozen `stage1_20260820_215228` run, including the assumed
reasoning-token overhead): 315 calls, ~672K input / ~10K visible-output /
~32K assumed-reasoning tokens, **$0.251 estimated, $0.502 worst-case** —
full report in `runs/ragas_20260821_094707/dry_run_estimate.json`. No
OpenAI API call was made to produce this; `OPENAI_API_KEY` was confirmed
absent from the shell environment before running the estimator.

## 2026-08-21 — RAGAS 6-sample pilot: two real bugs found and fixed, cost model corrected

Ran the real (paid) pilot: q01+q02 x arms A/B/C x 4 metrics = 24 judge
calls against gpt-5-mini, reasoning_effort=minimal, over the frozen
`stage1_20260820_215228` run. Run directory:
`runs/ragas_pilot_20260821_101134/`.

**Two real infrastructure bugs surfaced and were fixed before any
successful call completed** — this is exactly why the plan mandates a
pilot before the full run:

1. **Temperature rejection.** First attempt: all 24 calls failed with
   `400 Unsupported value: 'temperature' does not support 0.01 with this
   model`. Root cause: `eval/run_ragas.py` never set `temperature` on the
   `ChatOpenAI` object for reasoning models, but RAGAS's own
   `LangchainLLMWrapper.generate()` has a hardcoded `temperature=0.01`
   default it applies to every call regardless of how the wrapped
   langchain LLM was constructed — confirmed by reading
   `ragas.llms.base.LangchainLLMWrapper.agenerate_text`'s source directly.
   gpt-5-mini accepts only its default temperature (1). Fixed with RAGAS's
   own documented escape hatch: `LangchainLLMWrapper(..., bypass_temperature=True)`
   for reasoning-model judges. **No tokens were billed for this failure**
   — a 400 on an unsupported parameter is rejected before OpenAI processes
   any tokens, confirmed by inspecting the request-level error shape.
2. **Embeddings interface mismatch.** Second attempt: Faithfulness, Context
   Precision, and Context Recall all succeeded (18/24), but every
   `answer_relevancy` call failed with
   `AttributeError: 'HuggingFaceEmbeddings' object has no attribute 'embed_query'`.
   Root cause: `ragas.embeddings.HuggingFaceEmbeddings` (capital F) is
   ragas 0.4.3's newer `BaseRagasEmbedding` class (`embed_text()` only);
   the legacy `AnswerRelevancy` metric requires the older
   `BaseRagasEmbeddings` interface (`embed_query()`/`embed_documents()`).
   Fixed by wrapping `langchain_community.embeddings.HuggingFaceEmbeddings`
   (still local bge-small, still zero OpenAI embedding calls) in
   `ragas.embeddings.LangchainEmbeddingsWrapper`, which adapts any
   langchain `Embeddings` object to the interface `AnswerRelevancy` expects.
3. A third bug (accounting, not execution): `guard.record()` was
   unconditionally charging the pre-call token *estimate* against the
   budget even when a call failed every retry — meaning a run's reported
   `spent_usd` could include money that was never actually billed. Fixed
   to record spend only for `status == "ok"` results with real captured
   usage.

Stale error rows from these two runs remain in `scores.jsonl` (append-only
by design — see eval/run_generation.py's identical convention from Stage
1): 30 error rows plus the 24 final `status: "ok"` rows, one per
(question_id, arm, metric) triple, zero duplicates among the successful
set.

**Resume verified twice**: after the embeddings fix, re-running the exact
same pilot command skipped all 18 already-`ok` rows and attempted only the
6 previously-failing `answer_relevancy` calls (all succeeded, 0 retries).
Running the identical command again afterward produced `0 attempted, 24
skipped, 0 api calls, $0.00 spent` — confirmed no duplicate billing on
resume.

**Actual usage vs. dry-run estimate** (24 successful calls):

| | Estimated (pre-call) | Actual (measured) |
|---|---|---|
| Input tokens | — | 74,173 |
| Visible output tokens | — | 2,435 |
| Reasoning tokens | (100/call assumed) | **0** (every single call) |
| Total cost | $0.0370 | **$0.0234** |

The gap is almost entirely the reasoning-token assumption:
`REASONING_TOKENS_PER_CALL_ASSUMPTION` updated from 100 to **0**, based on
real measured evidence (all 24 calls, all 4 metrics, reported
`output_token_details.reasoning == 0`). Caveat carried into the code
comment: the pilot only covered 2 of 15 questions (both single-clause,
not multi-hop) — the full run should keep recording actual reasoning
tokens per call rather than assume 0 is guaranteed to hold everywhere.

**Corrected full-45 projection**: re-running the dry-run estimator with
the updated assumption gives **$0.1882** (worst-case $0.3764), which lines
up closely with the pilot-actual linear projection (`$0.0234 / 6 x 45` =
$0.1756) — the two independent methods now agree within ~7%, both
comfortably inside the $2-3 budget.

Budget guard did not trigger at any point across all three real
invocations (`stopped_early: false` every time).