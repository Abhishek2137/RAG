from pathlib import Path
from typing import Any, List

try:
    from langchain.schema import Document
except ImportError:  # langchain 1.x+ uses langchain_core
    from langchain_core.documents import Document

from pypdf import PdfReader


def load_all_documents(data_dir: str) -> List[Document]:
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents: List[Document] = []

    for text_file in sorted(data_path.glob("**/*.txt")):
        try:
            content = text_file.read_text(encoding="utf-8")
            documents.append(Document(page_content=content, metadata={"source": str(text_file)}))
            print(f"[DEBUG] Loaded text document: {text_file}")
        except Exception as exc:
            print(f"[ERROR] Failed to read text file {text_file}: {exc}")

    for pdf_file in sorted(data_path.glob("**/*.pdf")):
        try:
            reader = PdfReader(str(pdf_file))
            content_parts = []
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    content_parts.append(text)
            if content_parts:
                documents.append(Document(page_content="\n".join(content_parts), metadata={"source": str(pdf_file)}))
                print(f"[DEBUG] Loaded PDF document: {pdf_file} ({len(content_parts)} pages)")
            else:
                print(f"[DEBUG] PDF file {pdf_file} had no extractable text")
        except Exception as exc:
            print(f"[ERROR] Failed to read PDF file {pdf_file}: {exc}")

    print(f"[DEBUG] Loaded {len(documents)} documents from {data_path}")
    return documents

