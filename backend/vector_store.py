import os
from typing import List, Dict, Any
import chromadb # type: ignore
from chromadb.utils import embedding_functions # type: ignore

# Directory to persist vector database files locally
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Use a lightweight, open-source embedding model running locally
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

def index_chunks_in_vector_db(repo_name: str, chunks: List[Dict[str, Any]]) -> int:
    """Stores repository code chunks into a dedicated ChromaDB collection."""
    collection_name = repo_name.replace("/", "_").replace("\\", "_")
    
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=default_ef
    )

    documents = [c["page_content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"{c['metadata']['source']}_{c['metadata']['chunk_index']}" for c in chunks]

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.upsert(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    return collection.count()

def search_similar_code(repo_name: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Retrieves top-k most relevant code snippets from ChromaDB for a given query."""
    collection_name = repo_name.replace("/", "_").replace("\\", "_")
    
    try:
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=default_ef
        )
    except Exception:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    retrieved = []
    if results and results.get("documents"):
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        for doc, meta in zip(docs, metas):
            retrieved.append({"page_content": doc, "metadata": meta})

    return retrieved

# --- Aliases to match backend/main.py imports ---
def index_chunks(chunks: List[Dict[str, Any]], repo_url: str) -> str:
    """Alias adapter for main.py"""
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    index_chunks_in_vector_db(repo_name, chunks)
    return repo_name

def query_vector_db(repo_name: str, query: str) -> str:
    """Alias adapter for main.py that queries and returns a formatted string response."""
    results = search_similar_code(repo_name, query, top_k=3)
    if not results:
        return "No relevant code snippets found in this repository."
    
    context = "\n\n".join([f"File: {r['metadata']['source']}\n```\n{r['page_content']}\n```" for r in results])
    return f"Here are the relevant code sections found:\n\n{context}"