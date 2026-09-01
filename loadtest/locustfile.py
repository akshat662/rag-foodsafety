"""Load test for the FSSAI RAG QA API (app/api.py).

Two separate scenarios, as two separate Locust user classes, because
generation calls Groq at 30 RPM (see src/config.py's GENERATOR rate
limits): concurrent load against the full pipeline would 429 against
Groq's rate limiter almost immediately, measuring Groq's throughput cap
rather than this service's own retrieval/reranking pipeline.

  RetrievalOnlyUser  (Scenario A) -- POSTs with retrieval_only=true, so
      app/api.py never calls the generator and makes no external API call
      at all. Concurrency here measures OUR pipeline only: embedding,
      Chroma search, BM25, RRF, cross-encoder. Arm is fixed per run via the
      LOADTEST_ARM env var (see run_loadtest.sh, which runs this three
      times, once per arm).

  FullPipelineUser   (Scenario B) -- the real end-to-end path, including
      the Groq call. Self-terminates after exactly
      FULL_PIPELINE_MAX_REQUESTS (20) total requests and stops the whole
      runner -- intended to be run with exactly 1 user (see
      run_loadtest.sh), well inside Groq's 30 RPM limit, to measure
      realistic sequential end-to-end latency, not throughput.

Reads the API key from the APP_API_KEY environment variable -- never
hardcoded, matching this project's convention everywhere else a key is
needed (see app/auth.py).

Run with: locust -f loadtest/locustfile.py <UserClass> --headless ...
See loadtest/run_loadtest.sh for the exact commands and parameters used
for each scenario.
"""

import os
import random

from locust import HttpUser, constant, task
from locust.exception import StopUser

API_KEY = os.environ.get("APP_API_KEY")
if not API_KEY:
    raise RuntimeError("APP_API_KEY environment variable is not set")

# dense | hybrid | hybrid_rerank -- see app/schemas.py's Arm literal.
ARM = os.environ.get("LOADTEST_ARM", "dense")

HEADERS = {"X-API-Key": API_KEY, "Content-Type": "application/json"}

# ~10 casual dev/smoke questions covering the 3 ingested chapters (dairy,
# cereals, salt/spices/condiments) -- deliberately NOT data/qa_set.json's
# frozen 15 evaluation questions or data/refusal_set.json's frozen 5. These
# exist only to give retrieval something realistic to work over during a
# load test; nothing here is scored against ground truth.
DEV_QUESTIONS = [
    "What is the standard for pasteurized milk?",
    "How much fat does cream need to contain?",
    "What are the requirements for ghee?",
    "What is the moisture limit for wheat flour?",
    "What additives are permitted in bread?",
    "What is the standard for semolina?",
    "How much salt can iodized salt contain?",
    "What are the requirements for turmeric powder?",
    "What is the standard for black pepper?",
    "What is the composition standard for cardamom?",
]


class RetrievalOnlyUser(HttpUser):
    """Scenario A: retrieval_only=true, concurrent, no Groq call."""

    wait_time = constant(0)

    @task
    def query_retrieval_only(self):
        question = random.choice(DEV_QUESTIONS)
        self.client.post(
            "/query",
            json={"question": question, "arm": ARM, "retrieval_only": True},
            headers=HEADERS,
            name=f"/query [retrieval_only, arm={ARM}]",
        )


FULL_PIPELINE_MAX_REQUESTS = 20
_full_pipeline_request_count = 0


class FullPipelineUser(HttpUser):
    """Scenario B: full pipeline including generation.

    Run with exactly 1 user (see run_loadtest.sh) -- well inside Groq's
    30 RPM limit. Stops the whole runner after FULL_PIPELINE_MAX_REQUESTS
    total requests so the scenario is exactly "20 sequential requests",
    not an open-ended run that has to be manually timed out.
    """

    wait_time = constant(0)

    @task
    def query_full_pipeline(self):
        global _full_pipeline_request_count
        if _full_pipeline_request_count >= FULL_PIPELINE_MAX_REQUESTS:
            raise StopUser()

        _full_pipeline_request_count += 1
        question = random.choice(DEV_QUESTIONS)
        self.client.post(
            "/query",
            json={"question": question, "arm": ARM},
            headers=HEADERS,
            name=f"/query [full, arm={ARM}]",
        )

        if _full_pipeline_request_count >= FULL_PIPELINE_MAX_REQUESTS:
            self.environment.runner.quit()
