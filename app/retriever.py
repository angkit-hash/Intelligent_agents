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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "knowledge"
DB_DIR = Path(os.getenv("CHROMA_DB_DIR", ".chroma_db"))


def _load_documents() -> list[Document]:
    docs: list[Document] = []
    for file_path in sorted(DATA_DIR.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8")
        docs.append(Document(page_content=content, metadata={"source": file_path.name}))
    return docs


def build_vectorstore() -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=120)
    raw_docs = _load_documents()
    chunks = splitter.split_documents(raw_docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(DB_DIR),
        collection_name="agentic-rag-demo",
    )
    return vectorstore


def _normalize_text(text: str) -> list[str]:
    return [token.lower() for token in text.replace("\n", " ").split() if token.isalpha()]


def _keyword_overlap_score(question: str, chunk: str) -> float:
    question_tokens = Counter(_normalize_text(question))
    chunk_tokens = Counter(_normalize_text(chunk))
    overlap = sum(min(question_tokens[token], chunk_tokens[token]) for token in question_tokens)
    if overlap == 0:
        return 0.0
    return overlap / max(1, len(question_tokens))


def retrieve_and_rerank(question: str, vectorstore: Chroma, k: int = 5) -> list[Document]:
    hits = vectorstore.similarity_search(question, k=k)
    scored = []
    for doc in hits:
        score = _keyword_overlap_score(question, doc.page_content)
        scored.append((score, doc))

    scored.sort(key=lambda item: item[0], reverse=True)
    reranked = [doc for _, doc in scored if doc.page_content.strip()]
    return reranked[:3]
