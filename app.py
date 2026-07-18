import gradio as gr

from app.agent import run_agent


with gr.Blocks(title="Agentic RAG Demo") as demo:
    gr.Markdown(
        """
        # Agentic RAG Demo

        Ask a question about the sample invoice, resume, or support ticket set.
        The app performs a retrieval step, reranks the most relevant passages, and then answers using grounded evidence.
        """
    )

    question = gr.Textbox(
        label="Question",
        placeholder="Example: Which support ticket mentions a missing trailing slash?",
        lines=2,
    )
    submit = gr.Button("Run agent")
    answer = gr.Textbox(label="Answer", lines=8)

    submit.click(fn=run_agent, inputs=question, outputs=answer)


if __name__ == "__main__":
    demo.launch()
