from __future__ import annotations

from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from app.retriever import build_vectorstore, retrieve_and_rerank

load_dotenv()


class AgentState(TypedDict):
    question: str
    documents: list
    answer: str


def _build_vectorstore() -> object:
    return build_vectorstore()


def retrieve_node(state: AgentState) -> dict:
    vectorstore = _build_vectorstore()
    docs = retrieve_and_rerank(state["question"], vectorstore, k=5)
    return {"documents": docs}


def _dedupe_documents(documents: list) -> list:
    unique: list = []
    seen_sources: set[str] = set()

    for doc in documents:
        source = doc.metadata.get("source") if hasattr(doc, "metadata") else None
        if source and source in seen_sources:
            continue
        if source:
            seen_sources.add(source)
        unique.append(doc)

    return unique


def _grounded_summary(question: str, documents: list) -> str:
    if not documents:
        return "No relevant passages were retrieved for that question."

    question_tokens = {token.lower() for token in question.replace("\n", " ").split() if token.isalpha()}
    unique_docs = _dedupe_documents(documents)
    selected_parts: list[str] = []

    for doc in unique_docs:
        text = doc.page_content.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        matched_lines = [
            line for line in lines if any(token.lower() in line.lower() for token in question_tokens)
        ]
        if matched_lines:
            selected_parts.append("\n".join(matched_lines))
        else:
            selected_parts.append("\n".join(lines))

    return (
        "Grounded answer (evidence-based summary):\n\n"
        + "\n\n".join(selected_parts[:3])
    )


def answer_node(state: AgentState) -> dict:
    evidence = "\n\n".join(doc.page_content for doc in state["documents"])
    answer = _grounded_summary(state["question"], state["documents"])

    if not evidence.strip():
        answer = "No evidence was retrieved for that question."

    return {"answer": answer}


def run_agent(question: str) -> str:
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    app = graph.compile()
    result = app.invoke({"question": question, "documents": [], "answer": ""})
    return result["answer"]
