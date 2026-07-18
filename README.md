# RFP Response Assistant

An agentic RAG assistant that drafts RFP and security-questionnaire responses from previously won deals — citing the exact past answer, or flagging the question for review when no confident match exists.

Sales and solutions teams answer the same handful of questions over and over across every RFP and vendor security questionnaire ("Do you support SSO?", "What's your uptime SLA?", "Where is data stored?"). This assistant retrieves the closest previously answered question, drafts a response grounded in it, and cites which past deal it came from — or, if nothing in the knowledge base is a confident match, says so plainly instead of guessing at a compliance or security claim.

## How it works

1. **Retrieve** — the agent pulls the 5 most similar past answers from a local Chroma vector store.
2. **Rerank** — a keyword-overlap step (with common stopwords filtered out) reorders those candidates by literal relevance to the question.
3. **Confidence check** — if the best match doesn't clear a minimum relevance threshold, the agent flags the question for SME/proposal-team review instead of answering.
4. **Draft** — if the match is confident, a free, local Hugging Face model drafts a response using only the matched answer(s), citing the specific question ID and deal. Documents that rode along in retrieval but aren't individually close to the best match are excluded from the citation, so a tangential result can't get cited alongside the real answer.
5. **Fallback** — if the generation model isn't available or fails to load, a deterministic keyword-grounded summary of the matched answer is used instead, so the app always returns something grounded rather than failing outright.

## Project structure

```
.
├── app.py                # Gradio UI entry point
├── app/
│   ├── main.py             # CLI entry point
│   ├── agent.py            # LangGraph agent: retrieve -> confidence check -> draft/escalate
│   └── retriever.py        # Vector store creation, retrieval, and reranking
├── data/knowledge/        # Sample bank of previously answered RFP questions
├── .chroma_db/             # Local persisted vector store
├── requirements.txt
└── .env.example
```

## Why this architecture

### Chunking strategy

The knowledge base is split with `RecursiveCharacterTextSplitter` at roughly 600 characters with 120 characters of overlap — enough to keep one full past answer coherent in a single chunk, without losing context near a chunk boundary.

### Retrieval and reranking

Retrieval uses **Chroma** with open-source sentence embeddings (`all-MiniLM-L6-v2` by default), so the whole pipeline runs locally with no external API or per-call cost. Dense retrieval alone can surface passages that are semantically similar but not literally on-topic, so a second reranking pass scores candidates by keyword overlap with the question (stopwords excluded) and keeps only the closest matches.

### Confidence-gated answering

Two thresholds control how cautious the agent is:

- `MATCH_CONFIDENCE_THRESHOLD` (default `0.15`) — the floor the best match must clear before the agent will attempt an answer at all. Below this, the question is flagged for human review.
- `CITATION_RELATIVE_THRESHOLD` (default `0.6`) — once the agent decides to answer, only documents scoring within this fraction of the _best_ match are actually cited. This keeps a tangentially related result (e.g. an onboarding-timeline answer that happens to mention "SSO" in passing) from being cited alongside the real match just because it rode along in the same retrieval batch.

For security and compliance questions, a confidently wrong answer is worse than an honest "I don't know" — so the agent is built to know when to stay quiet.

## Getting started

### Prerequisites

- Python 3.11
- pip

### Installation

```bash
git clone https://github.com/angkit-hash/Intelligent_agents.git
cd Intelligent_agents
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and adjust values as needed:

```bash
cp .env.example .env
```

```dotenv
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
HF_GENERATION_MODEL=google/flan-t5-small
CHROMA_DB_DIR=.chroma_db
MATCH_CONFIDENCE_THRESHOLD=0.15
CITATION_RELATIVE_THRESHOLD=0.6
```

No API key is required — both models are free and run locally (the first run downloads them from Hugging Face).

### Run the app

```bash
python app.py
```

This launches the Gradio interface locally: paste in a new RFP or security-questionnaire question and get back a drafted, cited response or a review-flag notice.

Or run it from the command line:

```bash
python app/main.py "Do you support SSO and SAML for enterprise customers?"
```

## Adding your own past answers

Add a `.txt` file to `data/knowledge/` structured like the samples, with `Question ID:`, `Category:`, `Deal:`, `Question:`, and `Answer:` header lines, then rebuild the vector store by re-running the app (or calling `build_vectorstore()` directly).

## Limitations

- Reranking and confidence scoring are keyword-based, not semantic — a genuine match phrased very differently from the question could still score low and get escalated unnecessarily.
- `flan-t5-small` is a small, free model chosen for cost and latency, not reasoning depth. It's well suited to rephrasing already-matched evidence into a clean draft, but should not be trusted with questions that require combining or reconciling multiple sources.
- This tool should not be used unsupervised for pricing, legal liability, or contract-specific terms — those need human review regardless of the confidence score.

## Roadmap ideas

- Replace keyword-overlap reranking with a learned cross-encoder for semantic confidence scoring
- Swap in a hosted LLM provider as an optional, higher-quality generation backend
- Add a review queue / feedback loop so SME corrections become new knowledge-base entries
