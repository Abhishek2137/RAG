import re
from typing import List, Any

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover - fallback for older/newer LangChain layouts
    RecursiveCharacterTextSplitter = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - fallback when the package is unavailable
    SentenceTransformer = None

import numpy as np
from src.data_loader import load_all_documents


class SimpleChunk:
    def __init__(self, page_content: str, metadata: Any = None):
        self.page_content = page_content
        self.metadata = metadata or {}


class SimpleEmbeddingModel:
    def __init__(self, model_name: str = "fallback"):
        self.model_name = model_name

    def encode(self, texts: List[str], show_progress_bar: bool = False, normalize_embeddings: bool = False):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []
        for text in texts:
            tokens = re.findall(r"\b\w+\b", text.lower())
            if not tokens:
                vectors.append(np.zeros(32, dtype="float32"))
                continue

            vector = np.zeros(32, dtype="float32")
            for token in tokens:
                vector[abs(hash(token)) % 32] += 1.0

            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            vectors.append(vector)

        embeddings = np.stack(vectors, axis=0).astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        return embeddings


class SimpleTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Any]) -> List[Any]:
        chunks = []
        for document in documents:
            text = getattr(document, "page_content", str(document))
            words = text.split()
            start = 0
            while start < len(words):
                end = min(len(words), start + self.chunk_size)
                chunk_text = " ".join(words[start:end])
                chunks.append(SimpleChunk(chunk_text, getattr(document, "metadata", {})))
                if end == len(words):
                    break
                start = max(0, end - self.chunk_overlap)
        return chunks


class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = self._build_model(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def _build_model(self, model_name: str):
        if SentenceTransformer is None:
            print("[WARN] sentence-transformers unavailable; using lightweight fallback embeddings.")
            return SimpleEmbeddingModel(model_name)

        try:
            return SentenceTransformer(model_name)
        except Exception as exc:
            print(f"[WARN] Failed to load {model_name}: {exc}. Using lightweight fallback embeddings.")
            return SimpleEmbeddingModel(model_name)

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        if RecursiveCharacterTextSplitter is None:
            splitter = SimpleTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)