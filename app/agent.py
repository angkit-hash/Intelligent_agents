"""The agent graph: a sales RFP / security-questionnaire response assistant.

Given a new question from an incoming RFP or vendor security questionnaire,
this agent searches a bank of previously answered questions from won deals
(see data/knowledge/) and either:

  - finds a confidently matching past answer and drafts a response grounded
    in it, citing which past deal/question it's based on, or
  - decides no past answer is a close enough match, and flags the question
    for SME/proposal-team review rather than guessing at a compliance or
    security claim.

Built with LangGraph as a small two-node state machine:

    START -> retrieve -> answer -> END

The "retrieve" node builds/loads the vector store and pulls the most
relevant, reranked past answers for the question, each carrying a
relevance_score in its metadata. The "answer" node first checks whether the
best match clears MATCH_CONFIDENCE_THRESHOLD; if not, it returns a
review-flag message. If it does, it tries to generate a real draft with a
local, free Hugging Face model (configured via HF_GENERATION_MODEL); if that
model can't be loaded or run for any reason, it falls back to a
deterministic, keyword-grounded summary of the matched answer instead.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

from app.retriever import build_vectorstore, retrieve_and_rerank

load_dotenv()

# Minimum keyword-overlap relevance score (see retriever._keyword_overlap_score)
# the top match must clear before the agent will answer from it at all.
# Below this, the agent escalates instead of drafting a reply.
MATCH_CONFIDENCE_THRESHOLD = float(os.getenv("MATCH_CONFIDENCE_THRESHOLD", "0.15"))

# Once the agent has decided to answer, a document is only *cited* if its
# score is within this fraction of the best match's score. This is a
# separate, tighter bar than MATCH_CONFIDENCE_THRESHOLD: a tangentially
# related document (e.g. an onboarding-timeline answer that happens to
# mention "SSO" in passing) can clear the low absolute floor needed to
# avoid an unnecessary escalation, while still being far enough below the
# best match that it shouldn't be cited alongside it.
CITATION_RELATIVE_THRESHOLD = float(os.getenv("CITATION_RELATIVE_THRESHOLD", "0.6"))


class AgentState(TypedDict):
    """State passed between graph nodes."""

    question: str
    documents: list
    answer: str


def _build_vectorstore() -> object:
    """Thin wrapper so the graph node can be unit-tested/mocked easily."""
    return build_vectorstore()


def retrieve_node(state: AgentState) -> dict:
    """Graph node: build the vector store and fetch reranked evidence."""
    vectorstore = _build_vectorstore()
    docs = retrieve_and_rerank(state["question"], vectorstore, k=5)
    return {"documents": docs}


def _dedupe_documents(documents: list) -> list:
    """Drop chunks that come from a source file already represented.

    Keeps the answer from repeating the same document twice when multiple
    chunks from that document were retrieved.
    """
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


def _snippet_label(doc) -> str:
    """Build a human-readable citation like "RFP-041 (Security & Access
    Control, from Acme Corp)" from a document's parsed metadata, falling
    back to the filename if a snippet wasn't structured with the expected
    header fields.
    """
    question_id = doc.metadata.get("question_id")
    category = doc.metadata.get("category")
    deal = doc.metadata.get("deal")
    if question_id and category and deal:
        return f"{question_id} ({category}, from {deal})"
    if question_id and category:
        return f"{question_id} ({category})"
    if question_id:
        return question_id
    return doc.metadata.get("source", "Unknown source")


def _grounded_summary(question: str, documents: list) -> str:
    """Deterministic fallback: extract question-relevant lines from evidence.

    For each retrieved past answer, keep only the lines that share a word
    with the question (falling back to the whole answer if no line
    matches), label it with its citation, then join up to three answers
    together. This never calls a model, so it always works and always
    stays grounded in the retrieved text.
    """
    if not documents:
        return "No relevant past answers were retrieved for that question."

    question_tokens = {token.lower() for token in question.replace("\n", " ").split() if token.isalpha()}
    unique_docs = _dedupe_documents(documents)
    selected_parts: list[str] = []

    for doc in unique_docs:
        text = doc.page_content.strip()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        matched_lines = [
            line for line in lines if any(token.lower() in line.lower() for token in question_tokens)
        ]
        body = "\n".join(matched_lines) if matched_lines else "\n".join(lines)
        selected_parts.append(f"{_snippet_label(doc)}:\n{body}")

    return (
        "Grounded draft (based on past answers):\n\n"
        + "\n\n".join(selected_parts[:3])
    )


@lru_cache(maxsize=1)
def _get_generation_pipeline():
    """Lazily load the local Hugging Face generation model, once per process.

    Returns None if transformers/the model can't be loaded (e.g. missing
    dependency, no internet access to download weights, unsupported model
    name) so callers can gracefully fall back instead of crashing the app.
    """
    model_name = os.getenv("HF_GENERATION_MODEL", "google/flan-t5-small")
    try:
        from transformers import pipeline
    except ImportError:
        return None

    try:
        return pipeline("text2text-generation", model=model_name)
    except Exception:
        # Any failure to load (bad model name, no network, OOM, etc.) should
        # degrade gracefully rather than take down the whole app.
        return None


def _generate_answer(question: str, documents: list) -> str | None:
    """Try to generate an answer with the local model, grounded in evidence.

    Returns None (rather than raising) on any failure so the caller can fall
    back to the deterministic evidence summary instead.
    """
    generator = _get_generation_pipeline()
    if generator is None:
        return None

    unique_docs = _dedupe_documents(documents)
    context = "\n\n".join(f"{_snippet_label(doc)}:\n{doc.page_content.strip()}" for doc in unique_docs)
    if not context.strip():
        return None

    prompt = (
        "You are a proposal writer assistant helping answer a new RFP or security "
        "questionnaire question. Using only the past answer(s) in the context "
        "below, draft a polished response to the new question. Cite which past "
        "question ID and deal it's based on. If the context does not actually "
        "answer the question, say you don't have a verified answer.\n\n"
        f"Context:\n{context}\n\nNew question: {question}\nDraft response:"
    )

    try:
        result = generator(prompt, max_new_tokens=200)
        generated = result[0].get("generated_text", "").strip()
        return generated or None
    except Exception:
        return None


def answer_node(state: AgentState) -> dict:
    """Graph node: decide between drafting a response and flagging for review.

    First checks whether the best-retrieved past answer clears
    MATCH_CONFIDENCE_THRESHOLD. If it doesn't, the agent flags the question
    for SME/proposal-team review rather than drafting a response from a weak
    or irrelevant match — for security and compliance questions, a
    confidently wrong claim is far worse than admitting no verified answer
    was found. If the match is confident, generation is attempted first; if
    the model isn't available, fails to load, or returns nothing useful, the
    deterministic keyword-grounded summary is used instead so the app
    always returns a grounded, cited draft.
    """
    documents = state["documents"]

    if not documents:
        return {
            "answer": (
                "No matching past answer was found for this question. "
                "Flagging for SME/proposal-team review."
            )
        }

    best_score = max(doc.metadata.get("relevance_score", 0.0) for doc in documents)
    if best_score < MATCH_CONFIDENCE_THRESHOLD:
        return {
            "answer": (
                "No past answer closely matches this question "
                f"(best match confidence: {best_score:.2f}, below the "
                f"{MATCH_CONFIDENCE_THRESHOLD:.2f} threshold). "
                "Flagging for SME/proposal-team review rather than guessing."
            )
        }

    # Only cite documents that are close to the best match's confidence.
    # Without this, a weakly related document that merely rode along in the
    # top-k retrieval batch (e.g. an onboarding-timeline answer showing up
    # for an SSO question) would get cited as if it were relevant, just
    # because it cleared the low absolute floor needed to avoid escalating.
    citation_cutoff = best_score * CITATION_RELATIVE_THRESHOLD
    relevant_docs = [
        doc for doc in documents if doc.metadata.get("relevance_score", 0.0) >= citation_cutoff
    ]

    generated = _generate_answer(state["question"], relevant_docs)
    answer = generated if generated else _grounded_summary(state["question"], relevant_docs)

    return {"answer": answer}


def run_agent(question: str) -> str:
    """Build and run the retrieve -> answer graph for a single question."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "answer")
    graph.add_edge("answer", END)

    app = graph.compile()
    result = app.invoke({"question": question, "documents": [], "answer": ""})
    return result["answer"]
