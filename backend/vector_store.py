import hashlib
import os
import re
from typing import Any, Callable, Dict, List, Optional

import chromadb

CHROMA_PATH = os.getenv("CHROMA_PATH", os.path.join(os.getcwd(), "chroma_db"))
PREFIX = "rm_"
_HEADER = "# File: "

_client = None
_embed = None


def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client


def _get_embed():
    """Default Chroma embedder: all-MiniLM-L6-v2 on ONNX (no PyTorch needed)."""
    global _embed
    if _embed is None:
        from chromadb.utils import embedding_functions
        _embed = embedding_functions.DefaultEmbeddingFunction()
    return _embed


def set_embedding_function(fn) -> None:
    """Lets tests (or a hosted-embeddings backend) replace the embedder."""
    global _embed
    _embed = fn


def collection_name(repo_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]", "_", repo_id)[:40]
    digest = hashlib.md5(repo_id.encode()).hexdigest()[:8]
    return f"{PREFIX}{slug}_{digest}"


def _collection(repo_id: str, create: bool = False, repo_url: str = ""):
    client = _get_client()
    name = collection_name(repo_id)
    if create:
        return client.get_or_create_collection(
            name=name,
            embedding_function=_get_embed(),
            metadata={"repo_id": repo_id, "repo_url": repo_url, "hnsw:space": "cosine"},
        )
    return client.get_collection(name=name, embedding_function=_get_embed())


def _strip_header(doc: str) -> str:
    if doc.startswith(_HEADER) and "\n" in doc:
        return doc.split("\n", 1)[1]
    return doc


def index_chunks(
    repo_id: str,
    repo_url: str,
    chunks: List[Dict[str, Any]],
    progress: Optional[Callable[[int, int], None]] = None,
    batch_size: int = 128,
) -> Dict[str, int]:
    """Incremental index: only new/changed files are embedded; deleted files are purged."""
    col = _collection(repo_id, create=True, repo_url=repo_url)

    existing = col.get(include=["metadatas"])
    old_hashes: Dict[str, str] = {}
    for m in existing.get("metadatas") or []:
        if m and "source" in m and "file_hash" in m:
            old_hashes[m["source"]] = m["file_hash"]

    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        by_file.setdefault(c["metadata"]["source"], []).append(c)

    stats = {"added": 0, "updated": 0, "unchanged": 0, "removed": 0, "chunks_indexed": 0}
    to_add: List[Dict[str, Any]] = []

    for src, file_chunks in by_file.items():
        new_hash = file_chunks[0]["metadata"]["file_hash"]
        old = old_hashes.get(src)
        if old == new_hash:
            stats["unchanged"] += 1
            continue
        if old is not None:
            col.delete(where={"source": src})
            stats["updated"] += 1
        else:
            stats["added"] += 1
        to_add.extend(file_chunks)

    for src in old_hashes:
        if src not in by_file:
            col.delete(where={"source": src})
            stats["removed"] += 1

    total = len(to_add)
    for i in range(0, total, batch_size):
        batch = to_add[i:i + batch_size]
        col.add(
            documents=[f"{_HEADER}{c['metadata']['source']}\n{c['page_content']}" for c in batch],
            metadatas=[c["metadata"] for c in batch],
            ids=[f"{c['metadata']['source']}::{c['metadata']['chunk_index']}" for c in batch],
        )
        if progress:
            progress(min(i + batch_size, total), total)

    stats["chunks_indexed"] = total
    return stats


def search(repo_id: str, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    try:
        col = _collection(repo_id)
        n = col.count()
    except Exception:
        return []
    if n == 0:
        return []
    res = col.query(
        query_texts=[query],
        n_results=min(top_k, n),
        include=["documents", "metadatas", "distances"],
    )
    out = []
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    for doc, meta, dist in zip(docs, metas, dists):
        out.append({"page_content": _strip_header(doc), "metadata": meta, "distance": float(dist)})
    return out


def get_file_chunks(repo_id: str, source: str, limit: int = 3) -> List[Dict[str, Any]]:
    try:
        col = _collection(repo_id)
        res = col.get(where={"source": source}, include=["documents", "metadatas"])
    except Exception:
        return []
    rows = [
        {"page_content": _strip_header(d), "metadata": m, "distance": -1.0}
        for d, m in zip(res.get("documents") or [], res.get("metadatas") or [])
    ]
    rows.sort(key=lambda r: r["metadata"].get("chunk_index", 0))
    return rows[:limit]


def count(repo_id: str) -> int:
    try:
        return _collection(repo_id).count()
    except Exception:
        return 0


def get_repo_files(repo_id: str) -> List[Dict[str, Any]]:
    """One row per indexed file: source, ext, chunks, lines."""
    try:
        res = _collection(repo_id).get(include=["metadatas"])
    except Exception:
        return []
    files: Dict[str, Dict[str, Any]] = {}
    for m in res.get("metadatas") or []:
        row = files.setdefault(m["source"], {"source": m["source"], "ext": m.get("ext", ""), "chunks": 0, "lines": 0})
        row["chunks"] += 1
        row["lines"] = max(row["lines"], m.get("end_line", 0))
    return sorted(files.values(), key=lambda r: r["source"])


def list_repos() -> List[Dict[str, Any]]:
    client = _get_client()
    repos = []
    for c in client.list_collections():
        name = getattr(c, "name", c)
        if not str(name).startswith(PREFIX):
            continue
        try:
            col = client.get_collection(name=name, embedding_function=_get_embed())
        except Exception:
            continue
        meta = col.metadata or {}
        repos.append({
            "repo_id": meta.get("repo_id", name),
            "repo_url": meta.get("repo_url", ""),
            "chunks": col.count(),
        })
    return sorted(repos, key=lambda r: r["repo_id"].lower())


def delete_repo(repo_id: str) -> None:
    try:
        _get_client().delete_collection(collection_name(repo_id))
    except Exception:
        pass
