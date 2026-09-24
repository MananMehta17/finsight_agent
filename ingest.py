"""Build the FAISS index from everything in data/reports. Run once after adding reports."""
from tools import rag

if __name__ == "__main__":
    n_docs, n_chunks = rag.build_index()
    print(f"Indexed {n_docs} pages/files into {n_chunks} chunks -> vectorstore/")
