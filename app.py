"""Gradio front end for the RFP / security-questionnaire response assistant.

Launches a small web UI: a proposal or sales team member pastes in a new
RFP or security-questionnaire question, and the agent graph in
app/agent.py retrieves the most similar previously answered questions,
reranks them, and either drafts a grounded response citing the matching
past deal/question, or reports that no confident match was found and the
question should be flagged for SME review.
"""

import gradio as gr

from app.agent import run_agent


with gr.Blocks(title="RFP Response Assistant") as demo:
    gr.Markdown(
        """
        # RFP Response Assistant

        Paste in a new RFP or security-questionnaire question. The assistant searches
        previously answered questions from won deals, reranks the closest matches, and
        either drafts a response citing the matching past answer or tells you it found
        no confident match so the question can be flagged for SME/proposal-team review.
        """
    )

    question = gr.Textbox(
        label="New RFP / questionnaire question",
        placeholder="Example: Do you support SSO and SAML for enterprise customers?",
        lines=2,
    )
    submit = gr.Button("Draft answer from past responses")
    answer = gr.Textbox(label="Suggested draft / review flag", lines=8)

    submit.click(fn=run_agent, inputs=question, outputs=answer)


if __name__ == "__main__":
    demo.launch()
