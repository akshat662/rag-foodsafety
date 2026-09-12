# v2 Evaluation Report — Frozen 40-Question Benchmark + 10-Question Refusal Set

**Methodology note, read this first:** Stage 1 generation for this v2 pass used
**`gpt-4o-mini` (OpenAI)**, not the original Day 3 generator (**Qwen on
Groq**). The swap was made purely for throughput: Groq's free-tier account is
hard-capped at a measured 8,000 tokens/minute (see `decisions.md`,
2026-08-19), which made a 120-call batch take 30-40+ minutes; OpenAI's
standard tier completed the same 120 calls in well under a minute. The RAGAS
judge (`gpt-5-mini`) is unchanged from every prior run.

This means:
- **Retrieval metrics (Recall@K, MRR, complete/partial/miss) are generator-independent and directly comparable** to every previously published number (`reports/secondary_corpus_scale.md`, `reports/arm_comparison.csv`) — they never touch a generator at all.
- **Generation/RAGAS metrics (Faithfulness, Answer Relevancy, Context Precision, Context Recall) are NOT directly comparable across the old Qwen-based runs and this gpt-4o-mini-based run.** A generator swap and a benchmark-size change happened at the same time; changes here cannot be attributed solely to retrieval. Old-vs-new RAGAS comparisons below are done **within this single v2 pass only** (both the old 15 and new 25 were freshly regenerated with the same gpt-4o-mini generator), never against the historical Qwen numbers, except where explicitly marked "historical, different generator" for context.
- Generator and judge are both OpenAI models here (gpt-4o-mini vs gpt-5-mini — different generations, same vendor). This is a **disclosed limitation**, not a claim of eliminated bias: it reduces but does not eliminate self-preference risk relative to the original cross-vendor (Groq/Qwen vs OpenAI) design.

Artifacts backing this report: `eval/frozen_retrieval_eval_v2_40.json`,
`data/refusal_set_v2_10.json`, `runs/v2_40_large_openai/generations.jsonl`,
`runs/ragas_v2_40_large_openai/scores.jsonl`, `runs/refusal_v2_10_openai/refusal_results.jsonl`,
`reports/recall_v2_large_corpus.json`, `reports/ragas_v2_per_question.csv`.

---

## 1. Retrieval metrics (free, local, generator-independent)

All three arms, unchanged config (`k_final=3`, `candidate_k=20`, `rrf_k=60`), against the 263-chunk/9-chapter `fssai_large` corpus.

### 1.1 Single-clause questions (n=31 of 40)

| Arm | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| A — Dense | 0.935 | 0.968 | 1.000 | 1.000 | 0.960 |
| B — Hybrid RRF | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| C — Hybrid+rerank | 0.839 | 1.000 | 1.000 | 1.000 | 0.919 |

### 1.2 Multi-clause questions (n=9 of 40): complete / partial / miss at each pool depth

| Arm | k=1 complete | k=3 complete | k=5 complete | k=10 complete |
|---|---:|---:|---:|---:|
| A — Dense | 0.000 | 0.556 | 0.667 | 0.778 |
| B — Hybrid RRF | 0.000 | 0.667 | 0.889 | 1.000 |
| C — Hybrid+rerank | 0.000 | 0.778 | 0.889 | 1.000 |

(remaining probability mass at each k is "partial", not "miss" — no arm ever misses all required clauses for any multi-clause question by k=5.)

### 1.3 Old-15 vs. new-25 vs. all-40 (plain any-gold-clause-hit Recall@K, same methodology as `secondary_corpus_scale.md`)

| Subset | Arm | R@1 | R@3 | R@5 | MRR |
|---|---|---:|---:|---:|---:|
| Old 15 | A | 0.800 | 0.867 | 1.000 | 0.863 |
| Old 15 | B | 1.000 | 1.000 | 1.000 | 1.000 |
| Old 15 | C | 0.867 | 1.000 | 1.000 | 0.933 |
| New 25 | A | 0.920 | 0.960 | 1.000 | 0.950 |
| New 25 | B | 0.960 | 1.000 | 1.000 | 0.980 |
| New 25 | C | 0.840 | 1.000 | 1.000 | 0.920 |
| **All 40** | **A** | **0.875** | **0.925** | 1.000 | 0.918 |
| **All 40** | **B** | **0.975** | **1.000** | 1.000 | 0.988 |
| **All 40** | **C** | **0.850** | **1.000** | 1.000 | 0.925 |

The old-15 row here is a live re-run and matches `secondary_corpus_scale.md`'s published numbers exactly — confirms this v2 diagnostic reproduces the existing methodology correctly rather than introducing a silent scoring change.

**Reading**: Hybrid RRF's Recall@3 = 1.000 claim holds at n=40, on a harder, more diverse benchmark than the one it was originally measured on. Dense-only's Recall@3 (0.925 combined) is not catastrophically worse than before, but it is not 1.000 either, and its Recall@1 gap under Hybrid RRF (0.875 vs 0.975, a 10-point gap) is if anything *wider* at n=40 than the 0.800-vs-1.000 gap seen on the old 15 alone. **A genuinely unflattering, non-obvious result**: Hybrid+rerank's Recall@1 (0.850) is now *worse* than plain Dense's (0.875) across the full 40 — the reranker demotes some correct top-1 hits even as it reliably recovers them by k=3. This did not show up as clearly in the old 15-question benchmark and only became visible once the eval set diversified.

---

## 2. Generation / RAGAS metrics (gpt-4o-mini generator, gpt-5-mini judge; internal to v2 only)

### 2.1 Old-15 vs. new-25 vs. all-40, same generator throughout

| Subset | Arm | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---|---:|---:|---:|---:|
| Old 15 | A | 0.747 | 0.698 | 0.733 | 0.800 |
| Old 15 | B | 0.911 | 0.876 | 0.744 | 0.867 |
| Old 15 | C | 0.867 | 0.832 | 0.767 | 0.933 |
| New 25 | A | 0.873 | 0.830 | 0.773 | 0.803 |
| New 25 | B | 0.967 | 0.903 | 0.773 | 0.867 |
| New 25 | C | 0.920 | 0.938 | 0.737 | 0.893 |
| **All 40** | **A** | **0.826** | **0.781** | **0.758** | **0.802** |
| **All 40** | **B** | **0.946** | **0.893** | **0.763** | **0.867** |
| **All 40** | **C** | **0.900** | **0.898** | **0.748** | **0.908** |

### 2.2 Multi-clause subset only (n=9) — where retrieval completeness matters most

| Arm | Faithfulness | Answer Relevancy | Context Precision | Context Recall |
|---|---:|---:|---:|---:|
| A — Dense | 0.689 | 0.517 | 0.111 | 0.583 |
| B — Hybrid RRF | 1.000 | 0.788 | 0.056 | 0.667 |
| C — Hybrid+rerank | 1.000 | 0.927 | 0.222 | **0.889** |

Hybrid+rerank's advantage is concentrated almost entirely in multi-clause questions: its Context Recall (0.889) is far ahead of Dense (0.583) and Hybrid RRF (0.667) specifically where an answer needs evidence pulled from two different clauses. This directly reproduces the mechanism `secondary_corpus_scale.md` documented for q13/q14 (dense pulls only the cross-reference sentence, not the table with the actual numbers; reranker is the only arm that reliably retrieves both), now confirmed to generalize across 9 multi-clause questions, not just the original 2.

### 2.3 Abstention rate (all 40, gpt-4o-mini generator)

| Arm | Abstained |
|---|---:|
| A — Dense | 7/40 |
| B — Hybrid RRF | 2/40 |
| C — Hybrid+rerank | 2/40 |

Dense abstains roughly 3.5x more often than the other two arms — consistent with it having the weakest multi-clause completeness (an honest "I don't have enough context" is the correct response when retrieval genuinely failed, and 5 of Dense's 7 abstentions are exactly the multi-clause misses in Section 3 below).

### 2.4 Context vs. historical (Qwen/Groq generator, large corpus) — for orientation ONLY, not a valid comparison

| Arm | Faithfulness (Qwen, historical) | Faithfulness (gpt-4o-mini, old-15 subset, this run) |
|---|---:|---:|
| A | 0.850 | 0.747 |
| B | 1.000 | 0.911 |
| C | 0.900 | 0.867 |

The *ranking* (B > C > A) is preserved across both generators, but *absolute* Faithfulness drops under gpt-4o-mini for every arm by 0.03-0.10. This is presented only to show the pattern is somewhat generator-robust — **it is not evidence about retrieval quality changing**, since the generator changed too. Per the framing agreed for this evaluation: do not attribute this delta to retrieval.

---

## 3. Error analysis — every retrieval failure/partial at k=3 (production window)

10 of 120 (question, arm) pairs failed to retrieve complete gold evidence within the actual `k_final=3` window used by generation.

| Q | Type | Expected clause(s) | Retrieved top-3 (arm) | Failed arm(s) | Likely reason | Failure category |
|---|---|---|---|---|---|---|
| q13 | multi_hop | 2.1.6, 2.1.8 | A: 2.2.8-9, 2.2.4, 2.2.1 | A (miss) | Dense embedding pulled entirely into Ch.2.2 (fats/oils) — "Reichert Meissl"/"Polenske"/"Butyro-refractometer" vocabulary now exists there too since corpus expansion | semantic, cross-chapter |
| q13 | multi_hop | 2.1.6, 2.1.8 | B: 2.1.6, 2.2.8-9, 2.2.4 | B (partial) | Found the cross-reference clause (2.1.6) but not the ghee table (2.1.8) holding actual numbers; generator then answered with a vague cross-reference sentence, no numbers | multi-clause, chunking |
| q14 | multi_hop | 2.1.9, 2.1.8 | A: 2.2.1, 2.2.4, 2.2.4 | A (miss) | Same Ch.2.2 pull as q13 | semantic, cross-chapter |
| q14 | multi_hop | 2.1.9, 2.1.8 | B: 2.1.9, 2.2.4, 2.2.4 | B (partial) | Same as q13 pattern | multi-clause, chunking |
| q14 | multi_hop | 2.1.9, 2.1.8 | C: 2.2.1, 2.1.8, 2.2.1 | C (partial by clause-label, but generation-correct) | Reranker never retrieved 2.1.9 itself, only 2.1.8's ghee table — the generated answer was numerically correct anyway (2.1.8 alone has the values) but ungrounded in the actual cross-reference rule; a coincidence, not correct reasoning | ranking, generation (spurious correctness) |
| q16 | cross_chapter_confusion | 2.2.1, 2.1.8 | A, B: only 2.2.1 parts | A, B (partial, correctly hedged) | Same Ch.2.2-vs-Ch.2.1 Polenske collision as q13/q14 — this is the **same systemic weakness reproduced by a newly-authored question**, not a one-off | semantic, cross-chapter |
| q28 | negative_definition | 2.5.1 | A: 2.5.2 x3 | A (miss, correctly abstained) | Dense favored the large, topically-dominant 2.5.2 clause over the small adjacent 2.5.1 definitions clause | semantic, chunking (small-clause dilution) |
| q38 | sibling_disambiguation | 2.4.37, 2.4.38 | A: 2.4.1, 2.4.38, 2.4.17-18 | A (miss, correctly abstained) | Dense found one sibling (2.4.38) but drifted to other flour-family clauses instead of 2.4.37 | semantic, sibling confusion |
| q39 | cross_chapter_confusion | 2.9.6, 2.7.7 | C: 2.7.7, 2.10.3-4, 2.4.36-37 | **C (partial, and generated a confidently WRONG answer)** | The phrase "acid insoluble ash" / "ash insoluble in dilute HCl" recurs in >10 places across the corpus (spices, cocoa, chicory, yellow pea powder); reranker picked 2 lexically-similar decoys over the actually-relevant clove clause (2.9.6, true rank 10) | **ranking, lexical overload — most severe failure in this set** |

**q39/arm C is the single worst outcome in the entire 40-question benchmark**: the model did not abstain — it confidently stated "cocoa powder... vs yellow pea powder... making yellow pea powder the stricter limit," a fabricated comparison the question never asked about, sourced from a decoy chunk the reranker mistakenly preferred. Every other failure in this table degrades gracefully into either a correct-but-incomplete answer or a correct abstention; this one does not.

**No table-specific, chunking-boundary, or purely-generation-side failures were found beyond what's listed above** — every failure in this benchmark traces back to a retrieval miss/partial, consistent with the fact that the generation prompt (`src/generate.py`) explicitly instructs abstention when context is insufficient, and mostly obeys it (9 of 10 failure rows above either abstain or hedge correctly; only q39/C does not).

---

## 4. Refusal benchmark results (10 questions, gpt-4o-mini generator, large corpus)

| Arm | Correct abstentions | False answers |
|---|---:|---:|
| A — Dense | 10/10 | 0 |
| B — Hybrid RRF | 9/10 | 1 |
| C — Hybrid+rerank | 9/10 | 1 |

**Both false answers are the same question: r02 ("maximum moisture content in biscuits")** — a deliberately non-obvious refusal case designed to test whether an adjacent, genuinely-in-corpus topic (Chapter 2.4 cereal/flour moisture tables) would be mistaken for an answer to an out-of-corpus product (biscuits, which this corpus does not standardize at all).

- B retrieved clauses 2.4.6 (Food grains) and 2.4.14-15 (Starchy Foods) and answered: *"The maximum moisture content permitted in biscuits under FSSAI standards is 12.0 percent by mass."*
- C made the same error from the same clause pool.
- A (Dense) retrieved a different, less lexically-matched set of Ch.2.4 clauses and correctly abstained.

This is a real, reproducible false-positive: **12.0% is a genuine moisture limit in the corpus — for an unrelated grain/starch product, not biscuits** — and both Hybrid RRF and Hybrid+rerank presented it as if it were a biscuit standard. Dense's abstention here looks like a strength in this one case, but is consistent with Dense's generally weaker precision/confidence elsewhere in this evaluation, not a general claim that Dense is safer for refusal.

The other 9 refusal questions (including the 3 other "hard/non-obvious" cases — r03 infant formula, r05 nutraceuticals, r07 organic certification) were correctly abstained by all three arms.

---

## 5. Old vs. new benchmark: does the Hybrid-RRF-robustness claim still hold?

**Yes, on retrieval — the metric this claim was originally made about.** Recall@3 = 1.000 for Hybrid RRF holds at n=15 (original) and n=40 (this evaluation), across a corpus 2.1x larger and a question set that adds sibling-disambiguation, cross-chapter homonym collisions, and multi-clause questions the original 15 never tested. The dense-only degradation this claim was contrasted against also holds and is not worse than before (Recall@3 0.867 -> 0.925 combined, actually a slight *improvement* in the raw number, though this reflects the new 25 skewing easier for Dense than the adversarial q13/q14 pair, not that Dense actually got better).

**No, not unconditionally — two real qualifications surfaced only at n=40:**
1. Hybrid+rerank's Recall@1 (0.850) is now measurably *below* plain Dense's (0.875) across the full set — the original 15-question benchmark's C-arm Recall@1 gap (0.867, close to Dense's 0.800) did not make this direction of the tradeoff as visible.
2. The exact q13/q14 dense-vs-hybrid pattern reproduced itself on a newly-authored, independently-verified question (q16) that was written *before* any arm was run against it — this is the strongest evidence in this evaluation that the corpus-expansion-driven Ch.2.1/Ch.2.2 vocabulary collision is a systemic property of this corpus, not a two-question coincidence.

---

## 6. Final numbers (Task 9)

1. **Final question count**: 40 in-scope retrieval questions (`eval/frozen_retrieval_eval_v2_40.json`) + 10 out-of-scope refusal questions (`data/refusal_set_v2_10.json`), scored separately.
2. **Category distribution**: see `eval/frozen_eval_v2_summary.md` (9 chapters, 20%/35%/45% easy/medium/hard, 9/40 multi-clause).
3. **Dense**: Recall@3 (single-clause) 0.968, Recall@3 (all-40, any-hit) 0.925, multi-clause complete@3 0.556, RAGAS Faithfulness 0.826 / Context Recall 0.802, 7/40 abstentions.
4. **Hybrid RRF**: Recall@3 1.000 (both single-clause and all-40), multi-clause complete@3 0.667, RAGAS Faithfulness 0.946 / Context Recall 0.867, 2/40 abstentions.
5. **Hybrid+rerank**: Recall@3 1.000, multi-clause complete@3 0.778 (best of the three), but Recall@1 0.850 (worst of the three), RAGAS Faithfulness 0.900 / Context Recall 0.908 (best of the three), 2/40 abstentions.
6. **RAGAS (all 40, gpt-4o-mini generator)**: see Section 2.1 table; do not compare in absolute terms to the historical Qwen-generator numbers (Section 2.4).
7. **Refusal**: 10/10 (A), 9/10 (B), 9/10 (C); 1 reproducible false-answer case (r02, biscuits/cereal-table confusion), fully diagnosed in Section 4.
8. **Major failure modes**: (a) a real Ch.2.1-vs-Ch.2.2 vocabulary collision that degrades Dense specifically and reproduces on an independently-authored question, (b) multi-clause cross-references where the referring clause and the referenced table live far apart in the ranked pool, (c) one severe reranker failure (q39) where lexical overload of a common phrase ("acid insoluble ash") across the corpus caused a confidently wrong, non-abstained answer — the single worst outcome in the whole evaluation, (d) one refusal false-positive (r02) where an adjacent in-corpus numeric table was mistaken for an out-of-corpus product's standard.
9. **Is "Hybrid RRF improves robustness after corpus expansion" still supported?** Yes, on the metric it was originally claimed for (Recall@3), now validated at more than 2.5x the original question count and against newly-authored, pre-registered adversarial questions. It is not a universal win: Hybrid+rerank (not RRF) is the strongest arm on multi-clause completeness and RAGAS Faithfulness/Context Recall, while Hybrid RRF (not the reranker) has the best Recall@1. Reporting either arm as strictly "best" without qualification would overstate the evidence.
10. **CV/interview-ready claims**:
    - **Strong, keep**: "Built and froze a 40-question, ground-truth-verified retrieval benchmark across 9 regulatory chapters before running any experiment, explicitly to prevent evaluation leakage" — this is a real, demonstrable methodology practice, not a number.
    - **Strong, keep**: "Identified and reproduced (on an independently pre-written question) a systemic dense-retrieval failure mode caused by cross-chapter vocabulary collision, missed entirely by the original 3-chapter benchmark" — shows genuine error analysis, not just running an eval script.
    - **Needs qualifying, don't overstate**: "Hybrid RRF achieves Recall@3 = 1.000" — true and reproducible, but say "at k=3, on this corpus" rather than implying general superiority; Recall@1 and multi-clause completeness tell a more nuanced, arm-dependent story.
    - **Should be removed or reworded if currently claimed**: any CV line implying a single arm is unconditionally best across all metrics — the honest result is arm-dependent (RRF wins Recall@1, reranker wins multi-clause completeness and two of four RAGAS metrics).
    - **Do not claim**: that switching to a same-vendor (OpenAI-only) generator/judge pair "eliminates" self-preference bias in this evaluation — at most it is a design choice intended to reduce that risk relative to using the identical model for both roles, and this evaluation's generator swap (Qwen -> gpt-4o-mini, for rate-limit reasons) was a practical tradeoff, explicitly disclosed here, not a methodology upgrade.
