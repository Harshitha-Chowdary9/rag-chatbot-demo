"""Command-line chatbot over the documents in data/docs."""
import argparse

from rag.generator import answer
from rag.loader import load_chunks
from rag.retriever import Retriever


def main():
    ap = argparse.ArgumentParser(description="Chat with your documents (RAG demo)")
    ap.add_argument("--docs", default="data/docs", help="folder of .txt files")
    ap.add_argument("-k", type=int, default=3, help="chunks to retrieve")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    args = ap.parse_args()

    retriever = Retriever(load_chunks(args.docs))

    def ask(question: str):
        hits = retriever.search(question, k=args.k)
        print(answer(question, hits))
        for chunk, score in hits:
            print(f"   - {chunk.source} (chunk {chunk.index}, score {score:.2f})")

    if args.question:
        ask(args.question)
        return
    print("Ask a question (empty line to quit).")
    while (q := input("> ").strip()):
        ask(q)


if __name__ == "__main__":
    main()
