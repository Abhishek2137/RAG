from typing import Any, List

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:  # langchain 1.x+ uses langchain_text_splitters
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_documents(self, documents: List[Any]) -> np.ndarray:
        texts = [getattr(chunk, "page_content", str(chunk)) for chunk in documents]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings


if __name__ == "__main__":
    loader = load_all_documents("data/text_files")
    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(loader)
    embeddings = pipeline.embed_documents(chunks)
    print(f"[INFO] Generated embeddings for {len(chunks)} chunks.")
    print(f"[INFO] First embedding length: {embeddings.shape[1] if embeddings.size else 0}")