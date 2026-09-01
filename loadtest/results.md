# Load test results

**Environment**: single Docker container (`fssai-rag-api`, built from this
repo's `Dockerfile`), run locally on one machine — no network latency, no
load balancer, no multiple replicas. Host: Apple M3, 8 physical CPU cores,
16 GB RAM, macOS 15.6.1. Docker Desktop's VM (where the container actually
runs) was allocated 8 CPUs and ~7.75 GB RAM. CPU-only throughout — no GPU.
These numbers describe this one container on this one machine under this
one test, nothing more; see [Limitations](#limitations-of-this-test) below.

Tooling: [Locust](https://locust.io) 2.46.4 (`loadtest/locustfile.py`,
commands in `loadtest/run_loadtest.sh`). Raw CSVs: `loadtest/results/*.csv`.

## Scenario A — retrieval only (the real load test)

`retrieval_only: true` (see `app/schemas.py`) skips generation entirely —
no Groq call, no external API call of any kind. This measures only this
service's own pipeline: local embedding, Chroma similarity search, BM25,
Reciprocal Rank Fusion, and — for `hybrid_rerank` — the local cross-encoder
reranker. 10 concurrent users, 60 seconds, spawn rate 2 users/sec, one run
per arm.

| Scenario | Arm | Requests | Failures | p50 | p95 | p99 | RPS |
|---|---|---:|---:|---:|---:|---:|---:|
| A | dense | 4,797 | 0 | 120 ms | 140 ms | 160 ms | 81.22 |
| A | hybrid (dense+BM25+RRF) | 4,133 | 0 | 140 ms | 170 ms | 180 ms | 69.92 |
| A | hybrid + reranker | 54 | 0 | 9,800 ms | 10,000 ms | 10,000 ms | 0.93 |

Zero failures on all three arms — every request Chroma/BM25/RRF/the
reranker were asked to serve, they served correctly. But the gap between
`dense`/`hybrid` (p50 ~120–140 ms) and `hybrid + reranker` (p50 9.8
**seconds**, ~80x slower) is the single most important number this test
produced.

**Root cause: thread oversubscription, not a model-quality problem.** The
cross-encoder reranks the candidate pool by running one forward pass per
candidate — `candidate_k=20` (`src/config.py`), so 20 CPU-bound PyTorch
forward passes per request. PyTorch allocates multiple internal threads
per inference call by default. `app/api.py`'s `/query` handler is
synchronous, so FastAPI runs concurrent requests on separate threads —
with 10 concurrent requests each spinning up PyTorch's own multi-threaded
inference on top, on 8 CPU cores, those threads contend for cores rather
than run in parallel: far more threads are runnable than there is CPU to
run them on, so most of the "latency" is threads waiting on the scheduler,
not the reranker doing more work. Dense retrieval and BM25 are cheap
enough per-request (one embedding lookup, one lightweight lexical score,
neither spawning their own thread pools) that 10 concurrent threads don't
saturate the CPU the same way.

**Mitigation (not implemented here, out of scope for this test)**: cap
PyTorch's thread count per worker with `torch.set_num_threads(1)` so each
inference stops competing with itself across cores, and/or move the
reranker behind a bounded queue (a fixed-size worker pool or semaphore
around the cross-encoder call) instead of letting it run at per-request
concurrency — trading some queueing latency under load for predictable,
non-collapsing throughput. Neither change is made in this commit; this
is reported as a root cause and a direction, not a fix.

## Scenario B — full pipeline, sequential

Intended: `retrieval_only` omitted (default `false`), 1 user, 20 sequential
requests, well inside Groq's 30 RPM limit (`src/config.py`'s `GENERATOR`)
— measuring realistic end-to-end latency, not throughput.

**What actually happened**: this scenario did not complete. Across two
Locust runs (arm `hybrid_rerank`) and two follow-up isolated single calls
made with `curl` (no Locust, no concurrency, to rule out anything
Locust-specific) run to check whether the condition was transient, **12 of
13 full-pipeline attempts failed**, all with `RetryExhaustedError`
(`src/rate_limiter.py`: 5 retries exhausted against repeated 429s) or one
case of a malformed-JSON generation output (`BadRequestError`, a distinct,
non-rate-limit failure mode). A single isolated call made after a 90-second
cooldown still failed the same way. This is a live condition with the Groq
account used for this test, not an artifact of concurrency or of this
load-testing setup — the container's own local rate limiter believed it
had request/token budget available (per its configured 30 RPM / 8,000 TPM)
and paced calls accordingly; Groq's server rejected them anyway.

| Run | Requests | Failures | Failure modes |
|---|---:|---:|---|
| Locust run 1 (partial — killed by test-runner timeout) | 6 | 5 | 3x malformed-JSON generation, 2x retry-exhausted (429s) |
| Locust run 2 (fresh container/rate-limiter state) | 5 | 5 | 4x retry-exhausted (429s), 1x malformed-JSON generation |
| Isolated single calls (no Locust, incl. one after 90s cooldown) | 2 | 2 | 2x retry-exhausted (429s) |
| **Total this test session** | **13** | **12** | |

Failed-attempt latencies ranged **4.6 s to 134 s** — dominated entirely by
`src/rate_limiter.py`'s retry backoff (up to 5 attempts, exponential delay
capped at 60s) repeatedly hitting Groq's real rate limit, which is either
lower than this project's configured assumption (30 RPM / 8,000 TPM) at
this point in time, or otherwise degraded for this model/account when this
test was run. Not investigated further — diagnosing Groq's live account
state is outside this test's scope.

### Stage-level breakdown

Every failed attempt's own request log (`logs/requests.jsonl`) shows
retrieval completing quickly and successfully **even when generation
failed entirely** — the failure and almost all the latency is isolated to
the external call:

| Stage | Range across all 13 attempts (success and failure) |
|---|---|
| Retrieval (`retrieval_ms`, incl. reranking for `hybrid_rerank`) | 27 ms – 1,105 ms — succeeded on every single attempt |
| Generation (`generation_ms`) | 2,042 ms (the one success) to 133,890 ms (retry-exhausted failures) |

The one full-pipeline request that *did* succeed during this session
(captured earlier, before this rate-limit condition set in, arm `dense`):
retrieval 27.3 ms, generation 2,042.2 ms, total 2,069.6 ms — consistent
with the independently-validated number from this project's own Docker
smoke test (arm unspecified): retrieval 39.6 ms, generation 2,546.6 ms,
total ~2.6 s. On both of these clean data points, **generation is
~98–99% of total latency**, and that time is entirely the external Groq
API call — this service's own retrieval/reranking work is a rounding
error next to it, on the rare occasions generation succeeds at all.

**This is the finding Scenario A/B's split was designed to surface**: a
service whose own pipeline is fast and 100%-reliable (Scenario A: 0
failures across 8,984 requests) can still be nearly unusable end-to-end
(Scenario B: 12/13 failed) because generation depends on one external,
rate-limited, third-party call outside this project's control. It is also
exactly why Scenario A, not B, was designed as "the real load test" for
this system's own code.

## Limitations of this test

- Single container, one local machine, no network latency between load
  generator and server — not representative of a real deployment topology.
- Host CPU/RAM stated above; no claim about behavior on different
  hardware, more cores, or multiple container replicas.
- Scenario A's concurrency (10 users) and duration (60s) are arbitrary,
  reasonable-for-a-laptop values, not derived from any expected real
  traffic pattern — there is no production RPS or capacity target for this
  system, and none is implied by these numbers.
- Scenario B could not be completed as designed (20 successful sequential
  requests) because of a live external condition at test time (above);
  the failure itself, and the retrieval-stage evidence gathered around it,
  is reported instead of a synthetic clean run.
- The reranker's ~80x slowdown under 10-way concurrency is reported as
  observed, with a root cause (thread oversubscription — see Scenario A
  above) but no fix applied and no measurement at other concurrency
  levels, thread-count caps, or queueing designs.
- These are local, CPU-only, single-container numbers. Nothing here should
  be read as, or extrapolated into, a production throughput or capacity
  claim.
