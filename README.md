# RAG Chatbot Demo

A small retrieval-augmented generation (RAG) chatbot that answers questions from your own text documents and cites its sources.

It works fully offline out of the box (TF-IDF retrieval + extractive answers). Set `OPENAI_API_KEY` to have an LLM write the answers from the retrieved context.

## How it works
1. `rag/loader.py` splits each `.txt` file in `data/docs/` into overlapping word chunks.
2. `rag/retriever.py` builds a TF-IDF index (unigrams + bigrams) and returns the top-k chunks by cosine similarity, dropping low-scoring ones.
3. `rag/generator.py` builds a prompt from the chunks:
   - with `OPENAI_API_KEY`: asks the model to answer only from the context and cite `[file]`
   - without it: returns the best-matching sentences with their source
   - with no relevant chunks: says it doesn't know instead of guessing

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python chat.py                                   # interactive
python chat.py -q "What is predictive maintenance?"
python chat.py --docs path/to/your/txt/folder -k 5
export OPENAI_API_KEY=sk-...   # optional, enables LLM answers (RAG_MODEL to change model)
pytest
```
Add your own `.txt` files to `data/docs/` and restart. The sample docs cover industrial AI, predictive maintenance, visual inspection and RAG.

## Possible upgrades
Dense embeddings with a vector store (FAISS/Chroma), PDF loading, a web UI, retrieval evaluation (hit rate / MRR).
