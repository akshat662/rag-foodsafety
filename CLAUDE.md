# Project conventions
- Every retriever exposes: retrieve(query: str, k: int) -> list[Chunk]
- All LLM calls go through src/llm.py. Never call a provider SDK directly.
- All LLM calls go through the rate limiter. No exceptions.
- No new dependencies without asking. No LangGraph, no agents.
- All config lives in config.py. No hardcoded knobs.
- Eval must be resumable — cache per question on completion.
- data/qa_set.json is FROZEN. Never edit it.
- After any pipeline change, remind me to log it in DECISIONS.md.
- Write real commit messages. The git history is interview evidence.
