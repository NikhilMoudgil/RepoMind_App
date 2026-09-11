import os
import re
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

# Initialize persistent ChromaDB client
CHROMA_DATA_PATH = os.path.join(os.getcwd(), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Use default Sentence Transformer embedding model
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

def index_chunks_in_vector_db(repo_name: str, chunks: List[Dict[str, Any]]) -> int:
    """Incrementally updates ChromaDB by skipping unmodified files using file hashes."""
    # Sanitize collection name (ChromaDB allows only alphanumeric, underscores, hyphens)
    collection_name = f"repo_{re.sub(r'[^a-zA-Z0-9_-]', '_', repo_name)}"
    
    # Get or create collection without deleting existing data
    collection = chroma_client.get_or_create_collection(
        name=collection_name, 
        embedding_function=embedding_fn
    )

    # 1. Fetch existing metadatas from ChromaDB to build a hash lookup table
    existing = collection.get(include=["metadatas"])
    existing_hashes: Dict[str, str] = {}
    if existing and existing.get("metadatas"):
        for meta in existing["metadatas"]:
            if meta and "source" in meta and "file_hash" in meta:
                existing_hashes[meta["source"]] = meta["file_hash"]

    # 2. Group incoming chunks by source file relative path
    chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        source_path = c["metadata"]["source"]
        chunks_by_file.setdefault(source_path, []).append(c)

    new_chunks: List[Dict[str, Any]] = []

    # 3. Compare file hashes to detect new, modified, or unchanged files
    for file_path, file_chunks in chunks_by_file.items():
        current_hash = file_chunks[0]["metadata"].get("file_hash")
        indexed_hash = existing_hashes.get(file_path)

        # Skip file entirely if the hash matches existing record
        if indexed_hash == current_hash:
            continue

        # If file was modified, delete stale chunks from vector store first
        if indexed_hash is not None:
            collection.delete(where={"source": file_path})

        new_chunks.extend(file_chunks)

    # Return 0 immediately if no files changed
    if not new_chunks:
        return 0

    # 4. Prepare and insert only new or updated chunks
    documents = [c["page_content"] for c in new_chunks]
    metadatas = [c["metadata"] for c in new_chunks]
    ids = [f"{c['metadata']['source']}_chunk_{c['metadata']['chunk_index']}" for c in new_chunks]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return len(new_chunks)

def search_similar_code(repo_name: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Queries ChromaDB for code chunks most relevant to the query."""
    collection_name = f"repo_{re.sub(r'[^a-zA-Z0-9_-]', '_', repo_name)}"
    
    try:
        collection = chroma_client.get_collection(
            name=collection_name, 
            embedding_function=embedding_fn
        )
    except Exception:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        include=["documents", "metadatas"]
    )

    formatted_results = []
    if results and results.get("documents") and results["documents"][0]:
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            formatted_results.append({
                "page_content": doc,
                "metadata": meta
            })

    return formatted_results