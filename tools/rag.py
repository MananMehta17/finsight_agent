"""Tool 3: RAG over annual reports using LangChain + FAISS.

Flow: load PDFs/txt -> split into chunks -> embed each chunk -> store in FAISS.
At query time: embed the question -> find the TOP_K closest chunks -> return them
with their source file and page so the agent can cite them.
"""
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

_store = None   # cached in memory after first load


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


def load_documents(folder=config.REPORTS_DIR):
    docs = []
    docs += DirectoryLoader(str(folder), glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    docs += DirectoryLoader(str(folder), glob="**/*.txt", loader_cls=TextLoader,
                            loader_kwargs={"encoding": "utf-8"}).load()
    return docs


def build_index(folder=config.REPORTS_DIR, save_to=config.VECTORSTORE_DIR, embeddings=None):
    docs = load_documents(folder)
    if not docs:
        raise FileNotFoundError(f"No PDF or txt files found in {folder}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE,
                                              chunk_overlap=config.CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    store = FAISS.from_documents(chunks, embeddings or get_embeddings())
    store.save_local(str(save_to))
    global _store
    _store = store
    return len(docs), len(chunks)


def index_exists(path=config.VECTORSTORE_DIR) -> bool:
    return (path / "index.faiss").exists()


def load_index(path=config.VECTORSTORE_DIR, embeddings=None):
    global _store
    if _store is None:
        if not index_exists(path):
            raise FileNotFoundError("No index yet. Add reports to data/reports and run: python ingest.py")
        # FAISS metadata is stored with pickle. We built this file ourselves, so loading it is safe.
        # Never load an index file downloaded from someone you do not trust.
        _store = FAISS.load_local(str(path), embeddings or get_embeddings(),
                                  allow_dangerous_deserialization=True)
    return _store


def search(query: str, k: int = config.TOP_K) -> str:
    hits = load_index().similarity_search_with_score(query, k=k)
    blocks = []
    for i, (doc, score) in enumerate(hits, 1):
        src = doc.metadata.get("source", "unknown").split("/")[-1].split("\\")[-1]
        page = doc.metadata.get("page")
        where = f"{src}, page {page + 1}" if isinstance(page, int) else src
        # FAISS returns L2 distance: lower = more similar
        blocks.append(f"[{i}] ({where}, distance {score:.3f})\n{doc.page_content.strip()}")
    return "\n\n".join(blocks) if blocks else "No relevant passages found."
