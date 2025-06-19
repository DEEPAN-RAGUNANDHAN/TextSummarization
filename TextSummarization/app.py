import gradio as gr
from rag_summarizer import chunk_text, build_faiss_index, rag_summary

state = {"index": None, "chunks": None}

# ── Callbacks ─────────────────────────────────────────────────────────────────
def index_text(long_text):
    """
    Build the FAISS index for the provided long text.
    """
    try:
        if not long_text.strip():
            return "❌ Please paste some text."

        chunks = chunk_text(long_text)
        index, _ = build_faiss_index(chunks)

        state["index"], state["chunks"] = index, chunks
        return f"✅ Indexed {len(chunks)} chunks. Ready to summarize!"
    except Exception as e:
        print("Indexing error:", e)
        return f"❌ Indexing error: {e}"


def ask_question(query):
    """
    Generate a detailed RAG summary for the query.
    """
    if state["index"] is None:
        return "❌ Please index your text first (click 'Index Text')."

    try:
        return rag_summary(query, state["index"], state["chunks"])
    except Exception as e:
        print("Summarization error:", e)
        return f"❌ Summarization error: {e}"


# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 📝 RAG‑Based Detailed Text Summarizer")
    gr.Markdown("1. Paste your long text  →  2. Click **Index Text**  →  3. Ask for a summary")

    long_text = gr.Textbox(lines=18, label="Paste long text / script")
    index_btn = gr.Button("Index Text")
    status    = gr.Textbox(label="Status", interactive=False)

    index_btn.click(fn=index_text, inputs=long_text, outputs=status)

    query  = gr.Textbox(lines=2, label="Question or summary request", placeholder="e.g., Summarize key concepts")
    answer = gr.Textbox(lines=12, label="Detailed Summary")
    submit = gr.Button("Generate Summary")

    submit.click(fn=ask_question, inputs=query, outputs=answer)

if __name__ == "__main__":
    demo.launch()
