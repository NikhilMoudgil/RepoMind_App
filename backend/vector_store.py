import os
import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi  # type: ignore
import chromadb
from chromadb.utils import embedding_functions

# Initialize persistent ChromaDB client
CHROMA_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

# Default embedding function
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

def _tokenize_code(text: str) -> List[str]:
    """Tokenizes code by splitting camelCase, snake_case, and standard whitespace/punctuation."""
    # Split on non-alphanumeric characters and underscore
    tokens = re.findall(r'[A-Za-z0-9]+', text.lower())
    return tokens

def index_chunks_in_vector_db(repo_name: str, chunks: List[Dict[str, Any]]) -> int:
    """Stores code chunks in a ChromaDB collection for the given repository."""
    collection_name = f"repo_{re.sub(r'[^a-zA-Z0-9_-]', '_', repo_name)}"
    
    # Reset collection if it already exists
    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=collection_name, 
        embedding_function=embedding_fn
    )

    documents = [c["page_content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{idx}" for idx in range(len(chunks))]

    # Batch insert into ChromaDB
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    return len(chunks)

def reciprocal_rank_fusion(vector_results: List[Dict[str, Any]], bm25_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    """Merges vector search and BM25 search results using Reciprocal Rank Fusion (RRF)."""
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    # Score vector results
    for rank, doc in enumerate(vector_results):
        doc_id = f"{doc['metadata'].get('source')}_{doc['metadata'].get('chunk_index')}"
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

    # Score BM25 results
    for rank, doc in enumerate(bm25_results):
        doc_id = f"{doc['metadata'].get('source')}_{doc['metadata'].get('chunk_index')}"
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + (rank + 1)))

    # Sort documents by accumulated RRF score
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_doc_ids]

def search_similar_code(repo_name: str, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Performs Hybrid Search (BM25 + Dense Vector Embeddings) with Reciprocal Rank Fusion."""
    collection_name = f"repo_{re.sub(r'[^a-zA-Z0-9_-]', '_', repo_name)}"
    
    try:
        collection = chroma_client.get_collection(
            name=collection_name, 
            embedding_function=embedding_fn
        )
    except Exception:
        return []

    # 1. Retrieve dense vector matches from ChromaDB
    chroma_res = collection.query(query_texts=[query], n_results=min(10, collection.count()))
    vector_hits = []
    if chroma_res and chroma_res["documents"]:
        for doc, meta in zip(chroma_res["documents"][0], chroma_res["metadatas"][0]):
            vector_hits.append({"page_content": doc, "metadata": meta})

    # 2. Retrieve sparse BM25 matches across all documents in the collection
    all_docs = collection.get()
    if not all_docs or not all_docs["documents"]:
        return vector_hits[:top_k]

    corpus_docs = all_docs["documents"]
    corpus_metas = all_docs["metadatas"]

    tokenized_corpus = [_tokenize_code(doc) for doc in corpus_docs]  # type: ignore
    bm25 = BM25Okapi(tokenized_corpus)  # type: ignore

    tokenized_query = _tokenize_code(query)  # type: ignore
    bm25_scores = bm25.get_scores(tokenized_query)  # type: ignore

    # Rank top BM25 matches
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:10]
    bm25_hits = [
        {"page_content": corpus_docs[i], "metadata": corpus_metas[i]} 
        for i in top_bm25_indices if bm25_scores[i] > 0
    ]

    # 3. Combine using Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(vector_hits, bm25_hits)
    return fused_results[:top_k]