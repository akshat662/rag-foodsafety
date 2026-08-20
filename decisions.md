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