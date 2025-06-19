import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ── Models (loaded once) ───────────────────────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# You can swap in a larger model (e.g., facebook/bart-large-cnn, google/long-t5-tglobal-base)
summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
    device=-1,          # ‑1 ⇒ CPU (use 0 for GPU if available)
)

# ── Helper functions ──────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    """
    Break long text into overlapping chunks for embedding search.
    """
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def build_faiss_index(chunks):
    """
    Build an in‑memory FAISS index from text chunks.
    """
    emb = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    return index, chunks


def retrieve(query, index, chunks, k: int = 5):
    """
    Return top‑k most relevant chunks for the query.
    """
    q_emb = embedder.encode([query], convert_to_numpy=True)
    _, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0]]


def rag_summary(query, index, chunks, max_ctx_chars: int = 4000):
    """
    1. Retrieve context    2. Compose instruction prompt
    3. Generate summary    4. Return text
    """
    context = " ".join(retrieve(query, index, chunks))[:max_ctx_chars]

    prompt = (
        "Summarize the following content in a detailed, structured manner. "
        "Include all key concepts, critical data, and preserve the technical meaning. "
        "Use bullet points or sections where helpful.\n\n"
        + context
    )

    result = summarizer(
        prompt,
        max_length=400,   # up to ~512 tokens
        min_length=200,   # ensures it is reasonably long
        do_sample=False,
    )
    return result[0]["summary_text"]
