# RAGAS Evaluation — Arm Comparison and Failure Analysis

Source data (read-only, unmodified by this analysis):
- `runs/ragas_full_20260821_103101/scores.jsonl` — 180 metric evaluations (15 questions × 3 arms × 4 metrics), judge `gpt-5-mini` / `reasoning_effort=minimal`, evaluator commit `22b1d27812d43b21d3d2760d905219be4a7f3ac8`
- `runs/stage1_20260820_215228/generations.jsonl` — the frozen generation artifacts being judged, generator commit `a3cd8c0e3ffc02badd3925a9723c9d2b2ac0cdc8`

All 180 metric evaluations succeeded (0 errors, 0 retries needed). No code, retrieval, generation, or dataset changes were made to produce this analysis.

## 1. Headline result (all 15 questions)

| arm | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| A (dense) | 0.9778 | 0.9352 | 0.8111 | 1.0000 |
| B (hybrid RRF) | 0.9333 | 0.9342 | 0.8222 | 1.0000 |
| C (hybrid + rerank) | 0.9111 | 0.8620 | 0.7667 | 0.9333 |

**Contrary to the Day 3 plan's expected pattern** (A→B lifting context recall via BM25 catching lexical misses; B→C lifting context precision via the reranker filtering out sibling clauses), at n=15 **no metric improves monotonically from A→B→C**. All four metrics are flat-to-declining from A to C. Per the plan's own instruction (section 8), this is reported as the honest, observed result rather than hidden or explained away.

However, three-way ties dominate every pairwise comparison (12–15 of 15 questions tied per metric — see `arm_comparison.csv`, Section 2), and the differences that do exist below ~0.03 are noise per the plan's own threshold. The aggregate decline is not spread evenly across all 15 questions; the next section quantifies how much of it is concentrated in one already-known case.

## 2. Is the Arm C decline broad, or concentrated in q07?

q07 is the case flagged during Stage 1's manual eyeball review: the cross-encoder reranker selected the wrong split-part of clause 2.4.11 (a labelling section rather than the additive-limit table) plus two irrelevant clauses, causing arm C to correctly abstain rather than hallucinate an answer arm A/B did have context for.

Recomputing arm means with q07 excluded (n=14):

| arm | faithfulness | answer_relevancy | context_precision | context_recall |
|---|---|---|---|---|
| A | 0.9762 | 0.9306 | 0.8452 | 1.0000 |
| B | 0.9524 | 0.9295 | 0.8452 | 1.0000 |
| C | 0.9762 | 0.9235 | 0.8214 | 1.0000 |

**Context recall's entire A/B vs. C gap is q07.** Excluding it, all three arms score exactly 1.0 — q07 was the only question where retrieval missed a chunk needed to support the reference answer, and only for arm C.

**Faithfulness's apparent decline is also almost entirely q07.** Excluding it, arm C (0.9762) actually *ties* arm A and edges out arm B (0.9524) — the aggregate "C is less faithful" reading is not supported once the one abstention is set aside.

**Answer relevancy and context precision retain a smaller, real residual gap independent of q07** (answer_relevancy: A 0.9306 vs C 0.9235; context_precision: A 0.8452 vs C 0.8214) — driven by a handful of other individual questions (see Section 4), not a single outlier, though still within the plan's noise threshold at this sample size.

## 3. q07 deep dive — do the RAGAS metrics reflect the observed reranker failure?

**Yes, clearly, on the retrieval-side metric — and correctly not on the generation-side metrics, once the abstention is accounted for.**

| Arm | Retrieved clauses | Abstained | Answer | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---|---|---|---|---|---|
| A | 2.4.11, 2.1.19-20, 2.4.11 | No | "...1500 mg." | 1.0 | 1.0 | 0.333 | 1.0 |
| B | 2.4.11, 2.4.11, 2.4.6 | No | "...1500 mg per 100 g..." | 0.667 | 1.0 | 0.500 | 1.0 |
| C | 2.4.11, 2.4.19, 2.1.7 | **Yes** | "context does not provide enough information" | 0.0 | 0.0 | **0.0** | 0.0 |

- **Context Precision correctly detects the retrieval failure**: it is retrieval-only (question + reference vs. each retrieved chunk, independent of what the generator did with it) and drops to **0.0 for arm C** — none of C's three chunks (the wrong part of 2.4.11, Soybean 2.4.19, Cream 2.1.7) were judged useful for answering the Lecithins question, versus 0.333 (A) and 0.500 (B). This is exactly the retrieval regression Stage 1's manual review found, now confirmed quantitatively and independently by the judge.
- **Faithfulness, Answer Relevancy, and Context Recall all read 0.0 for arm C** — but this is a **metric artifact of the abstention, not evidence of ungrounded generation**. RAGAS's Faithfulness has no generated claims to check against context when the answer is "I don't know," Answer Relevancy scores a non-committal answer as unrelated to the question by design, and Context Recall has no supported statement to attribute. All three are downstream consequences of arm C correctly declining to answer from a context that did not contain the fact — which is the *safe*, intended behavior, not a grounding failure. Reading the raw 0.0s without this distinction would incorrectly rank arm C's generation as "worse" here, when the generator did exactly what the frozen system prompt asks (abstain when context is insufficient) and the actual fault is upstream, in retrieval.

This is precisely the retrieval-vs-generation distinction the evaluation was designed to support (see the Stage 2 kickoff's evaluation objective): **q07 is a retrieval failure, not a generation/faithfulness failure**, and Context Precision is the metric that correctly isolates it.

## 4. Other notable per-question patterns (independent of q07)

Not requested by name, but visible in `arm_comparison.csv` and worth surfacing rather than omitting:

- **q14** (multi-hop, Butter→Ghee cross-reference): Context Precision actually *improves* substantially for arm C (0.333 → 1.0), the one clear case in this dataset matching the plan's expected B→C precision-lift pattern — but Answer Relevancy is notably lower for arm C on the same question (0.778 vs. 0.932 for A). Worth a closer read if pursued further, but not analyzed beyond this observation here, per instructions not to tune or over-analyze.
- **q05, q12**: Context Precision drops for arm C (1.0 → 0.5 on both) — genuine reranker-selection differences, each a single sibling-disambiguation question where a plausible-but-wrong chunk apparently outranked the correct one after reranking.
- **q13, q15** (multi-hop): Context Precision is identical across all three arms (0.5 and 0.0 respectively) — a likely structural artifact of the metric judging each retrieved chunk's usefulness in isolation against a reference answer that requires combining two chunks, rather than a reranker effect. The Day 3 plan anticipated an analogous structural ceiling for Context Recall on multi-hop questions; this dataset's evidence suggests Context Precision can show the same ceiling for comparison-style multi-hop questions specifically. Flagged as a metric-methodology caveat, not a retrieval-arm difference (all three arms tie).

## 5. Bottom line

- The naive aggregate table (Section 1) suggests the reranker (arm C) underperforms across the board — but that reading is **not defensible without Section 2's q07 sensitivity check**, which shows most of the apparent decline (context recall entirely, faithfulness almost entirely) is one already-diagnosed retrieval failure, not a systematic reranker regression.
- A small residual gap in Answer Relevancy and Context Precision survives excluding q07 and is not explained by it — this is the more defensible version of "the reranker did not measurably improve retrieval quality on this corpus at n=15," which is itself a legitimate, reportable null result per the plan, rather than the stronger and less accurate claim that reranking made things worse.
- Context Precision is the metric that actually caught the q07 regression the eyeball review flagged; Faithfulness/Answer Relevancy/Context Recall all read 0.0 there purely as a consequence of a correct abstention, not as independent evidence of a generation problem.
