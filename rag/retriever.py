"""TF-IDF retriever with cosine similarity (no external services needed)."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .loader import Chunk


class Retriever:
    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("no chunks to index")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[tuple[Chunk, float]]:
        q = self.vectorizer.transform([query])
        scores = linear_kernel(q, self.matrix).ravel()
        order = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in order if scores[i] >= min_score]
