import pytest
from rag.generator import answer, extractive_answer
from rag.loader import Chunk, chunk_text, load_chunks
from rag.retriever import Retriever


def test_chunk_overlap():
    text = " ".join(str(i) for i in range(100))
    chunks = chunk_text(text, size=40, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]
    assert " ".join(chunks).count("99") >= 1


def test_chunk_invalid():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=5)


def test_empty_text():
    assert chunk_text("   ") == []


@pytest.fixture
def retriever():
    return Retriever(load_chunks("data/docs"))


def test_retrieves_right_document(retriever):
    hit, _ = retriever.search("How does predictive maintenance reduce downtime?")[0]
    assert hit.source == "predictive_maintenance.txt"
    hit, _ = retriever.search("Which CNN models detect surface defects?")[0]
    assert hit.source == "visual_inspection.txt"


def test_irrelevant_query_returns_nothing(retriever):
    assert retriever.search("banana smoothie recipe zzzz") == []


def test_answer_without_llm(retriever, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    hits = retriever.search("What is RAG?")
    out = answer("What is RAG?", hits)
    assert "rag_basics.txt" in out


def test_no_hits_says_unknown():
    assert "don't know" in extractive_answer("x", [])
