---
title: Agentic RAG Demo
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "6.20.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Agentic RAG Demo

This project is a compact, interview-friendly agentic retrieval-augmented generation (RAG) assistant. It answers questions over a small document set using a multi-step workflow:

1. The agent retrieves candidate passages from a local vector store.
2. A lightweight reranking step narrows the results.
3. A free open-source language model answers using the reranked evidence.

## Why this architecture

### Chunking strategy

The knowledge base is split with `RecursiveCharacterTextSplitter` at roughly 600 characters with 120 characters of overlap. This balances two goals:

- keep local context coherent for one business record or ticket
- avoid chunking too aggressively, which would lose important references and reduce answer quality

### Retrieval method

The app uses Chroma as the vector database and Hugging Face embeddings to create dense vector representations of each chunk. This is a good fit for a small demo because it is fast, local, and easy to inspect in the browser or terminal.

### Why rerank

Dense retrieval alone is often noisy. The rerank step gives a second signal by rewarding passages that are not only semantically close but also relevant to the literal question. That makes the final answer more grounded and less likely to hallucinate.

## Files

- `app/main.py` — runs the full demo
- `app/agent.py` — the agent graph and prompt orchestration
- `app/retriever.py` — vector store creation + retrieval + reranking
- `data/knowledge/` — sample documents such as an invoice, resume, and support ticket

## Run locally

```bash
pip install -r requirements.txt
copy .env.example .env
python app/main.py
```

No API key is required for the default free-model path.

## Deploy to Hugging Face Space

1. Push this repository to a GitHub repo.
2. Create a new Hugging Face Space.
3. Choose `Gradio` as the SDK.
4. Point the Space at the repo.
5. Keep the Space free-model path only; no API key is needed for the default deployment.

The default Space behavior uses Hugging Face-hosted free models for both embeddings and generation, which keeps the app portable and cheap to run.

The included `space.yaml` file tells Hugging Face to launch the Gradio app from `app.py`. The app is intentionally designed to run without any special server-only configuration.

If the local free-model generation backend is unavailable at runtime, the app gracefully falls back to a grounded evidence summary rather than hard-failing.
