import numpy as np
import streamlit as st

from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

st.set_page_config(page_title="RAG Demo", page_icon="📚", layout="centered")


@st.cache_resource(show_spinner=False)
def get_pipeline() -> EmbeddingPipeline:
    return EmbeddingPipeline(model_name="all-MiniLM-L6-v2", chunk_size=800, chunk_overlap=120)


@st.cache_data(show_spinner=False)
def build_index():
    documents = load_all_documents("data/text_files")
    pipeline = get_pipeline()
    chunks = pipeline.chunk_documents(documents)
    embeddings = pipeline.embed_chunks(chunks)
    return chunks, np.asarray(embeddings, dtype="float32")


def search_documents(query: str, top_k: int = 3):
    chunks, embeddings = build_index()
    if not chunks or embeddings.size == 0:
        return []

    pipeline = get_pipeline()
    query_embedding = pipeline.model.encode([query], normalize_embeddings=True).astype("float32")
    chunk_embeddings = embeddings.astype("float32")

    if chunk_embeddings.ndim == 1:
        chunk_embeddings = chunk_embeddings.reshape(1, -1)

    if chunk_embeddings.shape[1] != query_embedding.shape[1]:
        return []

    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    chunk_embeddings = chunk_embeddings / np.maximum(chunk_norms, 1e-12)
    query_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    query_embedding = query_embedding / np.maximum(query_norm, 1e-12)

    similarities = (chunk_embeddings @ query_embedding.T).ravel()
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append((float(similarities[idx]), chunks[int(idx)]))
    return results


st.title("RAG Demo")
st.write("Ask a question and the app will retrieve the most relevant passages from the local knowledge base.")

query = st.text_input("Question")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching the knowledge base..."):
            results = search_documents(query, top_k=3)

        if not results:
            st.info("No matching content was found.")
        else:
            for similarity, chunk in results:
                with st.expander(f"Similarity: {similarity:.2f}"):
                    st.write(getattr(chunk, "page_content", str(chunk)))