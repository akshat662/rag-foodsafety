#!/usr/bin/env bash
# Exact commands used to produce loadtest/results.md and the CSVs in
# loadtest/results/. Run against a single already-running container
# (see README.md "Running with Docker") -- this script does not build or
# start one itself.
#
# Requires: pip install -r requirements-loadtest.txt (locust; never
# installed into the served image -- see requirements-loadtest.txt and
# .dockerignore).
#
# Env vars this script expects to already be set:
#   APP_API_KEY   -- must match the value the running container was
#                    started with (`docker run -e APP_API_KEY=...`).
#   LOADTEST_HOST -- base URL of the running container, e.g.
#                    http://localhost:8000. Defaults to that if unset.

set -euo pipefail

HOST="${LOADTEST_HOST:-http://localhost:8000}"
OUT_DIR="$(dirname "$0")/results"
LOCUSTFILE="$(dirname "$0")/locustfile.py"

if [ -z "${APP_API_KEY:-}" ]; then
  echo "APP_API_KEY is not set -- export it to match the running container's key." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# ---------------------------------------------------------------------------
# Scenario A -- retrieval only, concurrent. No Groq call is made (see
# app/schemas.py's retrieval_only field), so this measures OUR pipeline:
# embedding, Chroma search, BM25, RRF, and (for hybrid_rerank) the
# cross-encoder reranker. Run once per arm, 10 concurrent users, 60s,
# spawn rate 2 users/sec.
# ---------------------------------------------------------------------------
for arm in dense hybrid hybrid_rerank; do
  echo "=== Scenario A: retrieval_only, arm=$arm ==="
  LOADTEST_ARM="$arm" locust -f "$LOCUSTFILE" RetrievalOnlyUser \
    --headless -u 10 -r 2 -t 60s -H "$HOST" \
    --csv "$OUT_DIR/scenario_a_${arm}" --only-summary
done

# ---------------------------------------------------------------------------
# Scenario B -- full pipeline (retrieval + generation), sequential. 1 user,
# 20 requests (the FullPipelineUser class stops the runner itself after 20
# -- see loadtest/locustfile.py), well inside Groq's 30 RPM limit. Measures
# realistic end-to-end latency, not throughput. Run once, against
# hybrid_rerank (the most complete pipeline: dense + BM25 + RRF + rerank +
# generation) -- see loadtest/results.md for why this scenario did not
# complete cleanly when it was actually run.
# ---------------------------------------------------------------------------
echo "=== Scenario B: full pipeline, arm=hybrid_rerank, 1 user, 20 requests ==="
LOADTEST_ARM="hybrid_rerank" locust -f "$LOCUSTFILE" FullPipelineUser \
  --headless -u 1 -r 1 -H "$HOST" \
  --csv "$OUT_DIR/scenario_b_hybrid_rerank" --only-summary

echo "Done. Results in $OUT_DIR/*.csv, written up in loadtest/results.md."
