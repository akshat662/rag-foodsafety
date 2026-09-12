# Frozen Evaluation v2 — Benchmark Composition

**Frozen at commit `d04db71077f8cfcedf0d50c484c894bb51d77e60`, before any retrieval or
generation experiment was run against these files.** No retrieval arm (Dense,
Hybrid RRF, Hybrid+rerank) was queried, and no chunk/embedding/reranker/prompt
setting was touched, while these files were being written. This document itself
was generated from the frozen JSON files' own fields (a short script counts
`chapter`, `difficulty`, `retrieval_type`, and `len(source_clauses)` across all
40/10 entries) — it is not hand-tallied, so it cannot silently drift from the
frozen data.

Files frozen by this pass:

- `eval/frozen_retrieval_eval_v2_40.json` — 40 in-scope retrieval questions
- `data/refusal_set_v2_10.json` — 10 out-of-scope refusal questions

---

## 1. Retrieval benchmark: `frozen_retrieval_eval_v2_40.json`

### 1.1 Provenance

- **Questions q01-q15**: byte-identical copies of the original
  `data/qa_set.json`. Not one field was changed, reworded, dropped, or
  re-graded. Verified programmatically at write-time (`assert a == b` against
  the live `data/qa_set.json`, see build script referenced in the run
  artifacts for this pass).
- **Questions q16-q40**: 25 new questions, authored directly from the actual
  FSSAI regulation text stored in the `fssai_large` Chroma collection
  (263 chunks / 9 chapters), never from a retriever's output or a model's
  generated answer. Every `source_clauses` entry in every one of the 40
  questions was mechanically checked against `fssai_large`'s real stored
  clause-label metadata (the same containment logic
  `scripts/ingest_large.py` uses for its own gold-clause check) — **zero
  missing or uncoverable clauses**.
- The original 15 were built and frozen against the 124-chunk/3-chapter
  primary corpus; several of them (q13, q14) are already known (from
  `reports/secondary_corpus_scale.md`) to be exactly where dense-only
  retrieval degrades once the corpus grows to 263 chunks. They are kept
  unchanged here specifically so that known result remains reproducible
  inside the larger 40-question set, not smoothed over.

### 1.2 Final count

**40 in-scope questions.** No out-of-scope/refusal questions are mixed into
this file.

### 1.3 Chapter distribution

Cross-chapter comparison questions count toward every chapter they cite.

| Chapter | Title | Questions touching it |
|---|---|---:|
| 2.1 | Dairy products and analogues | 8 |
| 2.2 | Fats, oils and fat emulsions | 6 |
| 2.3 | Fruit and Vegetable products | 5 |
| 2.4 | Cereals and Cereal products | 6 |
| 2.5 | Meat and Meat products | 3 |
| 2.6 | Fish and Fish products | 3 |
| 2.7 | Sweets and Confectionery | 4 |
| 2.9 | Salt, Spices, Condiments and related products | 5 |
| 2.10 | Beverages, Other than Dairy and Fruits & Vegetables based | 3 |

All 9 ingested chapters are represented by at least 3 questions; every
chapter added in the corpus expansion (2.2, 2.3, 2.5, 2.6, 2.7, 2.10) now has
direct benchmark coverage for the first time (the original 15 covered only
2.1, 2.4, 2.9).

### 1.4 Difficulty distribution

| Difficulty | Count | Share |
|---|---:|---:|
| Easy | 8 | 20% |
| Medium | 14 | 35% |
| Hard | 18 | 45% |

The new questions skew harder than the original 15 (which were 4 easy / 4
medium / 7 hard) — a deliberate consequence of leaning on sibling-product
tables, cross-chapter term collisions, and unit-normalization traps once a
larger, more heterogeneous corpus made those possible, not an attempt to
depress scores artificially.

### 1.5 Category / retrieval-type distribution

| Category | Count | What it probes |
|---|---:|---|
| `sibling_disambiguation` | 11 | Closely related products/variants sharing a table (task item 8) |
| `lexical` | 8 | Exact rare-term / numeric anchors, BM25-friendly (task item 3, 9) |
| `table_lookup` | 5 | Multi-row/multi-column table reading, limits & quantities (task item 6) |
| `cross_chapter_confusion` | 4 | Same term/test recurring in ≥2 chapters (task item 7) |
| `semantic` | 4 | Paraphrase avoids the regulation's own vocabulary (task item 2, 10) |
| `exception_condition` | 3 | Provisos, phased-in limits, named exceptions (task item 6) |
| `multi_hop` | 3 | Original-15 label for cross-clause reasoning (task item 5) |
| `unit_normalization_trap` | 1 | Same limit in two different units (task item 9, 10) |
| `negative_definition` | 1 | Explicit regulatory *exclusion* — hallucination-toward-affirmation trap |

(40 total; a question's single `retrieval_type` value reflects its dominant
character — several, e.g. q16 and q39, are simultaneously
cross-chapter-confusion AND sibling-disambiguation AND multi-clause; the
`notes` field on each question spells out every angle it covers, not just
the one used for this tally.)

### 1.6 Single- vs. multi-clause questions

| | Count |
|---|---:|
| Single-clause (one `source_clauses` entry, or several facts inside one stored chunk) | 31 |
| Multi-clause (2 distinct stored clauses/chunks required for a complete answer) | 9 |

The 9 multi-clause questions are: q13, q14, q15 (original) and q16, q20, q24,
q36, q38, q39 (new). For these, retrieval scoring must report **complete**
(all required clauses retrieved), **partial** (some but not all), or **miss**
(none) — plain Recall@K/Hit@K is not sufficient and is not used alone for
these 9 (see `eval/frozen_eval_v2_scoring.md` methodology, produced in the
next step of this evaluation).

### 1.7 Deliberately unbiased with respect to retrieval method

No question was included or excluded based on how Dense, Hybrid RRF, or
Hybrid+rerank performs on it — none of the three arms was run against any of
the 25 new questions before this file and this summary were written and
committed. The known q13/q14 dense-only weak spot from the original 15 was
**kept, not removed or rebalanced**, specifically so the 40-question set does
not artificially favor whichever arm happens to do best on brand-new
questions.

---

## 2. Refusal benchmark: `data/refusal_set_v2_10.json`

### 2.1 Provenance

The original `data/refusal_set.json` (5 questions: honey, edible oils, canned
fish, fruit jam, carbonated water) is **not reused as-is**. As
`reports/secondary_corpus_scale.md` already documented, corpus expansion to 9
chapters made 4 of those 5 topics (oils → 2.2, fish → 2.6, fruit/veg → 2.3,
beverages → 2.10) genuinely in-scope and answerable — keeping them as
"refusal" cases would now be **factually wrong test data**, not a stricter
benchmark. Only the honey question (Chapter 2.8, still never ingested) is
carried forward. The carbonated-water question's underlying fact was instead
converted into a real, verified in-scope retrieval question (q35 in the
40-question set above), since its ground truth (145 ppm caffeine) is now
directly checkable against the corpus.

### 2.2 Final count

**10 out-of-scope questions**, none of them mixed into the 40-question
retrieval set.

### 2.3 Category distribution

| Category | Count | Examples |
|---|---:|---|
| `plausible_but_unsupported` | 3 | r02 (biscuits), r03 (infant formula), r07 (organic certification) |
| `info_not_present` | 2 | r05 (nutraceutical dosage), r06 (licensing fees) |
| `medical_advice` | 2 | r08 (peanut allergy), r09 (cholesterol diet) |
| `food_related_not_covered` | 2 | r01 (honey), r04 (alcoholic beverages) |
| `clearly_unrelated` | 1 | r10 (WiFi router) |

### 2.4 Difficulty / obviousness distribution

| Difficulty | Count | Note |
|---|---:|---|
| Easy | 3 | r08, r09, r10 — no plausible false-positive retrieval path |
| Medium | 3 | r01, r04, r06 |
| Hard | 4 | r02, r03, r05, r07 — deliberately non-obvious: each has a real, topically-adjacent, **in-corpus** chunk that a weak retriever/generator could mistake for an answer (r02→Chapter 2.4 flour moisture tables, r03→Chapter 2.1 dairy-powder tables, r05/r06→regulations cross-referenced *by name* inside ingested chunks) |

Per the task instructions, this set intentionally does **not** make every
case obvious — 4 of the 10 are constructed so a shallow keyword/embedding
match on an unrelated-but-adjacent in-corpus chunk is a realistic failure
mode, not just a fully off-topic control question.
