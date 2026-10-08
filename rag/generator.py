"""Answer generation. Uses an LLM if OPENAI_API_KEY is set, otherwise an extractive fallback."""
import os
import re

from .loader import Chunk

PROMPT = (
    "Answer the question using only the context below. Cite sources as [file]. "
    "If the context does not contain the answer, say you don't know.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def build_context(hits: list[tuple[Chunk, float]]) -> str:
    return "\n\n".join(f"[{c.source}] {c.text}" for c, _ in hits)


def extractive_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """Pick the context sentences that share the most words with the question."""
    if not hits:
        return "I don't know - nothing relevant in the documents."
    q_words = set(re.findall(r"\w+", question.lower()))
    scored = []
    for chunk, _ in hits:
        for sent in re.split(r"(?<=[.!?])\s+", chunk.text):
            overlap = len(q_words & set(re.findall(r"\w+", sent.lower())))
            if overlap:
                scored.append((overlap, sent, chunk.source))
    scored.sort(key=lambda x: -x[0])
    top = scored[:2] or [(0, hits[0][0].text[:300], hits[0][0].source)]
    return " ".join(f"{s} [{src}]" for _, s, src in top)


def llm_answer(question: str, hits: list[tuple[Chunk, float]], model: str | None = None) -> str:
    from openai import OpenAI  # imported lazily so the demo works without it

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model or os.environ.get("RAG_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": PROMPT.format(context=build_context(hits), question=question)}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    if not hits:
        return "I don't know - nothing relevant in the documents."
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return llm_answer(question, hits)
        except Exception as exc:  # network/key problems should not kill the demo
            return f"(LLM unavailable: {exc})\n" + extractive_answer(question, hits)
    return extractive_answer(question, hits)
