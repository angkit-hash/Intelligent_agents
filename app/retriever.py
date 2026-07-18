"""Vector store construction, retrieval, and reranking for the agentic RAG demo.

This module is responsible for everything that happens *before* an answer is
generated:

1. Loading the sample documents from ``data/knowledge/``.
2. Splitting them into overlapping chunks.
3. Embedding those chunks and storing them in a local Chroma vector store.
4. Given a question, retrieving the most similar chunks and reranking them.

The reranking here is intentionally simple (keyword/token overlap) rather than
a learned cross-encoder model, so the whole pipeline stays fast, local, and
easy to reason about for a demo.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Location of the sample knowledge base (invoice, resume, support ticket, etc).
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"

# Where the persisted Chroma database lives. Configurable via .env so you can
# point different environments (tests, demos) at different DB directories.
DB_DIR = Path(os.getenv("CHROMA_DB_DIR", ".chroma_db"))


def _parse_snippet_fields(content: str) -> dict:
    """Pull structured "Field: value" lines (Question ID, Category, Deal)
    out of a past-answer file's leading lines, so they can be attached as
    metadata instead of just living inside the raw text.

    This lets the agent cite "RFP-041, from the Acme Corp deal" directly
    rather than just pointing at a filename.
    """
    fields = {}
    field_map = {
        "question id": "question_id",
        "category": "category",
        "deal": "deal",
    }
    for line in content.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        if key in field_map:
            fields[field_map[key]] = value.strip()
    return fields


def _load_documents() -> list[Document]:
    """Load every .txt file in the knowledge directory into a Document.

    Each document's filename is stored in metadata under "source". Past-answer
    files also get "question_id", "category", and "deal" parsed out of their
    header lines (see _parse_snippet_fields) so later steps can cite the
    specific past RFP answer a chunk came from, e.g. in agent.py.
    """
    docs: list[Document] = []
    for file_path in sorted(DATA_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        metadata = {"source": file_path.name}
        metadata.update(_parse_snippet_fields(content))
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_vectorstore() -> Chroma:
    """Build (or rebuild) the local Chroma vector store from the sample docs.

    Chunking uses a 600-character window with 120 characters of overlap,
    which keeps each chunk large enough to contain a coherent record (e.g.
    one invoice or one support ticket) while still overlapping enough that
    references near a chunk boundary aren't lost.

    Embeddings come from a small, free sentence-transformers model
    (all-MiniLM-L6-v2 by default) so the whole demo runs locally without any
    API key or paid service.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    raw_docs = _load_documents()
    chunks = splitter.split_documents(raw_docs)

    embedding_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="agentic-rag-demo",
    )
    return vectorstore


# Common words excluded from keyword-overlap scoring. Without this, a
# question like "what is the weather today" scores artificially high against
# any ticket, since "the"/"is"/"what" appear everywhere regardless of topic —
# which would defeat the confidence threshold used to decide when to
# escalate instead of answering.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "if", "then", "so", "to", "of", "in", "on", "at",
    "for", "with", "from", "by", "as", "that", "this", "these", "those",
    "it", "its", "what", "when", "where", "why", "how", "do", "does", "did",
    "can", "could", "should", "would", "will", "shall", "i", "you", "we",
    "they", "he", "she", "our", "your", "their", "not", "no", "get", "gets",
}


def _normalize_text(text: str) -> list[str]:
    """Lowercase and tokenize text into meaningful alphabetic words.

    Punctuation, numbers, and whitespace are dropped so that keyword overlap
    scoring compares words on equal footing (e.g. "path." and "path" match).
    Common stopwords are also dropped so scoring reflects topical overlap
    rather than shared grammar.
    """
    tokens = [token.lower() for token in text.replace("\n", " ").split() if token.isalpha()]
    return [token for token in tokens if token not in _STOPWORDS]


def _keyword_overlap_score(question: str, chunk: str) -> float:
    """Score how much a chunk's vocabulary overlaps with the question's.

    This is a cheap stand-in for a real reranker model: it counts how many
    question words also appear in the chunk (using a multiset/Counter so
    repeated words count proportionally), normalized by the number of unique
    question tokens. Higher score = more literal keyword relevance.
    """
    question_tokens = Counter(_normalize_text(question))
    chunk_tokens = Counter(_normalize_text(chunk))
    overlap = sum(min(question_tokens[token], chunk_tokens[token]) for token in question_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(question_tokens))


def retrieve_and_rerank(question: str, vectorstore: Chroma, k: int = 5) -> list[Document]:
    """Retrieve top-k similar chunks, then rerank by keyword overlap.

    Dense similarity search alone can surface passages that are
    "semantically close" but not literally relevant to the question. This
    second pass reorders the candidates by how many of the question's actual
    words appear in each chunk, then returns only the top 3 — the ones most
    likely to directly ground an answer.

    Each returned document has its relevance score attached at
    metadata["relevance_score"], so callers (e.g. agent.py) can decide
    whether the best match is confident enough to answer from, or whether
    the question should be escalated to a human instead.
    """
    hits = vectorstore.similarity_search(question, k=k)

    scored = []
    for doc in hits:
        score = _keyword_overlap_score(question, doc.page_content)
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    reranked = []
    for score, doc in scored:
        if not doc.page_content.strip():
            continue
        doc.metadata["relevance_score"] = score
        reranked.append(doc)

    return reranked[:3]
