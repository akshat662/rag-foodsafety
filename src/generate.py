"""Grounded RAG generation over retrieved chunks.

Builds a prompt that restricts the model to the supplied context, calls the
configured generator through src.llm's uniform interface, and returns a
structured {answer, citations, abstained} result. No retrieval, evaluation,
or direct provider SDK calls happen here.
"""

from dataclasses import dataclass

from src.config import GENERATOR
from src.llm import get_llm
from src.retrieval.vector import Chunk

_ABSTENTION_MESSAGE = "The supplied context does not provide enough information to answer this question."

_SYSTEM_PROMPT = """You are a regulatory question-answering assistant for FSSAI food safety regulations.

Answer ONLY using information explicitly contained in the supplied context chunks. Do not use outside knowledge or your own parametric/trained knowledge of food regulations.

Do not infer, estimate, or extrapolate a regulatory limit, value, or requirement that is not explicitly stated in the supplied chunks, even if it seems reasonable or you recall it from training.

If the supplied chunks do not contain enough information to answer the question, you must explicitly abstain: set "abstained" to true and state in "answer" that the supplied context does not provide enough information to answer the question.

Cite the clause number(s) that support your answer in "citations", using exactly the clause numbers shown in the context (e.g. "2.4.1").

Preserve all numeric values, units, percentages, ppm values, and conditions exactly as they appear in the supplied text. Do not round, convert, or paraphrase them.

Respond with a single JSON object with exactly these fields: "answer" (string), "citations" (array of strings), "abstained" (boolean).

Always provide the answer as a complete, concise sentence.
Include the relevant unit whenever a numeric value is given.
Do not output a bare number or isolated value.
"""

_ANSWER_SCHEMA = {
    "answer": "string",
    "citations": "list[string]",
    "abstained": "bool",
}


@dataclass(frozen=True)
class GeneratedAnswer:
    """A grounded generation result: the answer, its supporting clause
    citations, and whether the model abstained due to insufficient context.
    """

    answer: str
    citations: list[str]
    abstained: bool


def generate(question: str, chunks: list[Chunk]) -> GeneratedAnswer:
    """Generate a grounded answer to `question` using only `chunks` as context.

    Routes through src.llm.get_llm (never a provider SDK directly) using the
    configured generator (src.config.GENERATOR: groq / openai/gpt-oss-20b).
    If `chunks` is empty there is nothing to ground an answer in, so this
    abstains immediately without calling the model.
    """
    if not chunks:
        return GeneratedAnswer(answer=_ABSTENTION_MESSAGE, citations=[], abstained=True)

    llm = get_llm(GENERATOR.provider, GENERATOR.model)
    prompt = _build_prompt(question, chunks)
    raw = llm.generate_structured(prompt, schema=_ANSWER_SCHEMA, system=_SYSTEM_PROMPT)

    return _parse_answer(raw)


def _build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Combine the retrieved chunks into a labeled context block plus the question."""
    context = _format_context(chunks)
    return f"Context (retrieved regulation excerpts):\n\n{context}\n\nQuestion: {question}"


def _format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks into a context block labeled with clause/chapter/page
    so the model can cite them accurately.
    """
    blocks = [
        f"[Clause {chunk.clause}] {chunk.heading} (Chapter {chunk.chapter}, page {chunk.page})\n"
        f"{chunk.text}"
        for chunk in chunks
    ]
    return "\n\n---\n\n".join(blocks)


def _parse_answer(raw: dict) -> GeneratedAnswer:
    """Coerce the model's raw structured-output dict into a GeneratedAnswer."""
    abstained = bool(raw.get("abstained", False))
    citations = [str(c) for c in raw.get("citations", []) if c]
    answer = str(raw.get("answer", "")).strip()
    if abstained and not answer:
        answer = _ABSTENTION_MESSAGE
    return GeneratedAnswer(answer=answer, citations=citations, abstained=abstained)
