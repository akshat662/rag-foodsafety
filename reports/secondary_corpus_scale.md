# SECONDARY EXPERIMENT: does dense-retrieval saturation survive corpus scale?

**This is a secondary experiment. It does not supersede the primary frozen
result** (3-arm ablation over a 124-chunk, 3-chapter corpus, recall@3 =
1.00 for all arms, reported separately). Every retrieval parameter below
(embedding model, chunking logic, `k_first_stage` = 20, `k_final` = 3,
RRF `k` = 60, generation prompt, temperature = 0.0) is identical to the
primary experiment. The same frozen 15-question benchmark and clause-level
ground truth (`data/qa_set.json`) is used, unmodified. The only variable is
corpus size.

**Question**: was the primary experiment's recall@3 = 1.00 for dense-only
retrieval a genuine property of dense retrieval on this domain, or an
artifact of a corpus small enough that every gold clause was trivially the
single best embedding match?

## Step 1 — Ingestion

New, separate Chroma collection `fssai_large` at `data/chroma_large/`
(distinct from the primary `fssai_regulations` collection at
`data/chroma/`, which was not touched). Built by `scripts/ingest_large.py`,
reusing `src.chunking.chunk_document` unchanged and the same embedding
model (`BAAI/bge-small-en-v1.5`) as the primary experiment.

Source: 9 FSSAI compendium chapter PDFs in `data/raw/` — the original 3
(2.1, 2.4, 2.9) plus 6 new ones (2.2, 2.3, 2.5, 2.6, 2.7, 2.10).

| Chapter | Title | Chunks |
|---|---|---:|
| 2.1 | Dairy products and analogues | 37 |
| 2.2 | Fats, oils and fat emulsions | 23 |
| 2.3 | Fruit and Vegetable products | 61 |
| 2.4 | Cereals and Cereal products | 44 |
| 2.5 | Meat and Meat products | 14 |
| 2.6 | Fish and Fish products | 22 |
| 2.7 | Sweets and Confectionary | 8 |
| 2.9 | Salt, Spices, Condiments and related products | 43 |
| 2.10 | Beverages, Other than Dairy and Fruits & Vegetables based | 11 |
| **Total** | | **263** |

The three original chapters reproduced **identically** to the primary
collection's chunk counts (37 / 44 / 43 = 124), confirming chunking is
deterministic and unaffected by this ingestion path.

**Caveat**: 263 chunks is ~2.1x the primary corpus's 124, not the ~4x
originally anticipated — the 6 additional chapters are individually
shorter than 2.1/2.4/2.9 (2.7 alone contributes only 8 chunks). The corpus
did get materially larger and more heterogeneous, which is what this
experiment needs, but the scale-up is smaller than planned; worth knowing
before reading the recall deltas below as "4x-scale" evidence.

**Gold clause check**: all 14 distinct clauses referenced by
`data/qa_set.json`'s `source_clauses` are present in `fssai_large`'s
metadata (verified against merged-range clause labels too, e.g. gold
`2.4.4` is covered by stored label `2.4.4-2.4.5`). No missing clauses —
did not need to fail loudly.

## Step 2 — Local recall@k / MRR diagnostic (zero API calls)

`scripts/recall_at_k_large.py`. Each arm retrieves a top-20 ranked pool
per question (`RETRIEVAL.candidate_k`, unchanged); recall@k is scored at
prefix lengths 1/3/5/10 against clause-level ground truth, MRR over the
same pool.

| Arm | Corpus | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| A — dense only | small (124) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| A — dense only | large (263) | 0.800 | 0.867 | 1.000 | 1.000 | 0.863 |
| A — dense only | **delta** | **-0.200** | **-0.133** | +0.000 | +0.000 | **-0.137** |
| B — hybrid RRF | small (124) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B — hybrid RRF | large (263) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| B — hybrid RRF | **delta** | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |
| C — hybrid+rerank | small (124) | 0.867 | 1.000 | 1.000 | 1.000 | 0.933 |
| C — hybrid+rerank | large (263) | 0.867 | 1.000 | 1.000 | 1.000 | 0.933 |
| C — hybrid+rerank | **delta** | +0.000 | +0.000 | +0.000 | +0.000 | +0.000 |

### Reading

- **Dense-only retrieval is not scale-invariant.** At ~2.1x corpus size,
  recall@1 drops 20 points (1.00 → 0.80) and recall@3 drops 13 points
  (1.00 → 0.867) — 2 of 15 questions no longer have the gold clause as the
  single best dense match, and 2 no longer have it in the top 3. Recall@3
  is still the operative metric (`k_final` = 3), so **this is the number
  that matters**: dense-only's primary-experiment recall@3 = 1.00 was, at
  least partly, an artifact of corpus size, not a ceiling effect intrinsic
  to dense retrieval on this domain. The 2 recall@3 misses are **q13** and
  **q14** — both multi-hop questions whose ground truth spans two clauses
  (2.1.6/2.1.9 plus 2.1.8's ghee table). On the large corpus, dense-only's
  top-3 for both is pulled entirely into Chapter 2.2 (Fats, oils and fat
  emulsions) — a new, topically-adjacent chapter whose fat-quality language
  ("Reichert Meissl", "Polenske", "Butyro-refractometer" all appear there
  too) out-competes the correct dairy clauses once it exists in the corpus.
  This is a concrete instance of the failure mode "small corpus saturation
  hides", not just a statistical artifact: a real, topically-adjacent
  distractor chapter that the primary 3-chapter corpus had no way to
  surface.
- **Hybrid RRF (arm B) is untouched by the scale-up** — recall@3 = 1.00 on
  both corpora, MRR = 1.00 on both. The BM25 lexical component appears to
  be fully compensating for whatever dense retrieval is missing at this
  scale. This is a positive result for hybrid retrieval specifically: the
  primary experiment's finding that "hybrid had no headroom over dense" was
  a consequence of the corpus being too small to separate the two arms —
  at slightly larger scale, dense alone degrades and hybrid does not.
  Whether that gap widens further with more corpus growth is untested here.
- **Hybrid+rerank (arm C) is also untouched**, and identical to the small
  corpus at every k — including recall@1 = 0.867, which was already below
  1.00 on the small corpus (the reranker demotes the gold chunk out of
  first place for the same 2 questions on both corpora). Reranking here
  neither helps nor hurts relative to hybrid RRF on either corpus.
- All three arms still reach recall@5 = recall@10 = 1.000 on the large
  corpus — the gold clause is never pushed out of a 5-chunk window even
  where dense-only's top-1/top-3 degrades.

## Step 3 — Stage 1 generation + Stage 2 RAGAS

Run `runs/secondary_large_20260901_122742/` (config.json, git_commit.txt,
generations.jsonl, scores.jsonl, per_question.csv, aggregate.json).
git commit `6eef7f8` (the "add optional collection params" commit — Step 3
did not run against uncommitted code). All three arms, all 15 questions.

**Stage 1 (generation):** 45/45 pairs succeeded, 0 errors, 0 retries.
Abstentions: A (dense) 3/15, C (hybrid+rerank) 1/15, B (hybrid-RRF) 0/15.

**Stage 2 (RAGAS):** 180/180 (question, arm, metric) scores succeeded, 0
errors. Judge: gpt-5-mini, reasoning_effort=minimal (same as primary).
Actual cost: **$0.1605** (dry-run estimate was $0.191; primary-experiment
Stage 2 pilot data had already shown estimates run conservative — see
decisions.md).

### RAGAS comparison — small corpus vs. large corpus, full 15 questions

| Arm | Corpus | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---:|---:|---:|---:|
| A — dense | small | 0.978 | 0.935 | 0.811 | 1.000 |
| A — dense | large | 0.850 | 0.745 | 0.722 | 0.800 |
| A — dense | **delta** | **-0.128** | **-0.190** | **-0.089** | **-0.200** |
| B — hybrid-RRF | small | 0.933 | 0.934 | 0.822 | 1.000 |
| B — hybrid-RRF | large | 1.000 | 0.916 | 0.733 | 0.867 |
| B — hybrid-RRF | **delta** | **+0.067** | -0.018 | **-0.089** | **-0.133** |
| C — hybrid+rerank | small | 0.911 | 0.862 | 0.767 | 0.933 |
| C — hybrid+rerank | large | 0.900 | 0.871 | 0.733 | 0.933 |
| C — hybrid+rerank | **delta** | -0.011 | +0.009 | -0.034 | +0.000 |

Bold deltas are above the ~0.03 noise floor this project treats as
meaningful at n=15 (per the primary Day 3 plan: "differences below ~0.03
at n=15 are noise"); plain deltas (B's answer_relevancy, C's faithfulness/
answer_relevancy/context_recall, and C's context_precision borderline at
-0.034) should be read as noise, not signal.

`aggregate.json` also reports both tables restricted to the
`parametric_known == false` subset (n=12) — the same ranking holds there;
not reproduced here for brevity.

### q13 / q14 — the two dense recall@3 misses, per-arm RAGAS scores

| Question | Arm | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Abstained |
|---|---|---:|---:|---:|---:|---|
| q13 | A | 1.00 | 0.00 | 0.00 | **0.00** | **yes** |
| q13 | B | 1.00 | 0.79 | 0.00 | **0.00** | no |
| q13 | C | 1.00 | 0.92 | 0.50 | **1.00** | no |
| q14 | A | 0.25 | 0.00 | 0.00 | **0.00** | **yes** |
| q14 | B | 1.00 | 0.79 | 0.00 | **0.00** | no |
| q14 | C | 1.00 | 0.96 | 0.50 | **1.00** | no |

This is the retrieval deficit propagating into answer quality, in three
different ways per arm:

- **A abstains** — correctly refuses rather than hallucinate, but produces
  no answer at all (context_recall/precision/relevancy all 0 by
  construction: there is no answer to score against ground truth).
- **B does NOT abstain, and looks fine on faithfulness (1.00) and answer
  relevancy (~0.79) — but context_recall is 0.00 for both.** B retrieved
  the primary clause (2.1.6 / 2.1.9) but never retrieved 2.1.8 (the ghee
  table holding the actual Reichert Meissl / Polenske / Butyro-refractometer
  numbers), so its answer paraphrases the cross-reference ("must meet the
  standards... as prescribed for ghee") without ever stating a number.
  Faithfulness scores high because nothing in that vague answer
  contradicts the retrieved context — but it's faithful to the wrong,
  incomplete context. This is exactly the kind of miss recall@k alone
  can't see (B "hit" the recall@3 target for its own retrieved-clause
  definition) but RAGAS's reference-based context_recall catches
  immediately.
- **C is the only arm that retrieves both clauses and scores context_recall
  = 1.00 on both**, producing complete, numeric answers matching ground
  truth.

### This is a SECONDARY experiment

**This does not supersede the primary frozen result.** The primary
3-chapter/124-chunk ablation (recall@3 = 1.00 for all arms, RAGAS scores
in `reports/ragas_summary.csv`) remains the headline, reportable result.
This experiment exists only to characterize whether that saturation
generalizes — and the answer, on this evidence, is "not for dense-only."

**Scale caveat, restated**: the corpus grew ~2.1x (124 → 263 chunks), not
the ~4x originally planned — and the dense-only degradation above (recall@3
1.00 → 0.867, RAGAS context_recall 1.00 → 0.80) already appears at that
more modest 2.1x scale. Whether it worsens, plateaus, or is an artifact of
which specific chapters were added (2.2's fat-quality vocabulary
specifically collides with q13/q14's dairy-fat vocabulary) is untested —
this result should be read as "saturation is not scale-invariant," not as
a calibrated scale-vs-recall curve.

**Refusal set note**: `data/refusal_set.json` (5 out-of-scope questions —
honey, edible oils, fish, fruit/veg, beverages) was frozen and validated
at 15/15 correct abstentions (5 questions x 3 arms) against the
**primary** 3-chapter corpus, where all 5 topics are genuinely out of
scope. That is no longer true for `fssai_large`: chapters 2.2
(edible oils/fats), 2.3 (fruit/vegetable), 2.6 (fish), and 2.10 (beverages)
are now ingested, so most of the refusal set's questions have become
genuinely in-scope for this collection. The primary experiment's abstention
result applies only to the primary corpus. Refusal was **not** re-run or
re-reported against `fssai_large`, per instruction — `data/refusal_set.json`
was not read or modified in this experiment.
