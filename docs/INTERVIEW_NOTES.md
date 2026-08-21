# Interview Notes

Prepared answers for the questions this project is likely to draw. Each one
is meant to be said out loud in a few sentences, not read verbatim.

## 1. Why RAG instead of fine-tuning?

Regulations change frequently. Retrieval lets the knowledge base update
(re-ingest the new corpus) without retraining a model, and it gives every
answer a citation back to the specific clause it came from — fine-tuning
bakes facts into weights with no traceable source and no cheap update path.

## 2. Why hybrid retrieval (dense + BM25)?

Dense embeddings handle semantic similarity — a paraphrased question that
shares no vocabulary with the source clause. BM25 handles exact tokens
dense embeddings tend to under-weight: clause numbers, INS additive codes,
chemical names. The two catch different failure modes, so combining them
covers more ground than either alone.

## 3. Why RRF (Reciprocal Rank Fusion) specifically?

Because BM25's score and cosine similarity aren't on the same scale — one
is an unbounded lexical score, the other a bounded distance. Averaging or
weighting them directly would require an arbitrary normalization. RRF
sidesteps that by fusing on *rank* (1st, 2nd, 3rd place in each list)
rather than the raw scores, so the two retrievers never need to be
directly compared.

## 4. Why add a reranker on top of that?

Cross-encoders (joint attention over the query and a candidate's full
text) are more accurate than embedding similarity but too expensive to run
over the whole corpus. The standard pattern is: retrieve cheaply first
(dense + BM25 + RRF down to a candidate pool of 20), then rerank only that
smaller pool with the expensive model, before cutting to the final top 3.

## 5. Why didn't reranking improve the numbers here?

Because there wasn't much room left to improve. The recall@k diagnostic
shows dense retrieval alone already achieved recall@3 = 1.0 — the correct
clause was in the top 3 for every one of the 15 evaluation questions
before hybrid retrieval or reranking were applied at all. On a benchmark
this size and this corpus (124 chunks, 3 chapters), there was no
missing-document problem left for either later stage to fix. That's a
statement about this benchmark's headroom, not a general claim that
hybrid retrieval or reranking don't work.

## 6. Walk me through the q07 case.

q07 asks for the Lecithins (INS 322) limit in a children's supplement. All
three arms retrieved the correct clause, 2.4.11. Dense and hybrid also
happened to retrieve the specific chunked *part* of that clause containing
the actual limit table; the reranker instead promoted a different part of
the same clause — its labelling section — plus two unrelated clauses.
Recall@k and RAGAS's Context Precision both confirm this: the clause was
retrieved, the right sub-chunk wasn't. Because the system's grounding
prompt only allows answering from the supplied context, it correctly
abstained rather than guess or fall back to parametric knowledge. This is
the strongest concrete example in the whole evaluation of a
chunk-fragmentation limitation: a clause split into multiple parts is
sometimes reassembled incorrectly by reranking, even though the base
retrieval found the right document. It is not evidence that reranking is
broadly ineffective, and it is not a hallucination — it's a retrieval
selection issue with a correct downstream response.

## 7. How do you know you didn't tune anything based on the evaluation results?

The 15 evaluation questions and 5 refusal questions were frozen and
committed to git before any retrieval, reranking, generation, or RAGAS
scoring ran against them — the commit hash is recorded in `decisions.md`
alongside the freeze. Every subsequent generation and evaluation run
records the git commit its results came from, so a run whose commit
doesn't match the frozen state would be visibly invalid. Retrieval
parameters (`k_final=3`, `candidate_k=20`) were also committed before
Stage 1 generation began, and nothing in `src/retrieval/` changed after
that commit through the end of evaluation.

## 8. Is this production-grade?

No. It's containerized, API-served, logged, and evaluated — that's a
specific, honest claim about what's actually been built and validated. It
does not have monitoring/alerting, CI/CD, load testing, or an SLA, and the
evaluation benchmark is small (15 questions) and narrow (3 chapters, 124
chunks). Calling it production-grade would overclaim what's actually
there; "production-shaped" is the more accurate description.
