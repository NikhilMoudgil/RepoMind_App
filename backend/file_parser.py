import os
import hashlib
from typing import Any, Dict, Iterable, List, Tuple

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter  # type: ignore

IGNORED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist", "build",
    ".vscode", ".idea", ".next", "target", "vendor", ".mypy_cache", ".pytest_cache",
    "site-packages", "chroma_db", "cloned_repos",
}
IGNORED_FILES = {"package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock"}

# extension -> (label shown in the UI, language name for syntax highlighting)
LANGUAGES = {
    ".py": ("Python", "python"), ".js": ("JavaScript", "javascript"), ".jsx": ("JavaScript", "jsx"),
    ".ts": ("TypeScript", "typescript"), ".tsx": ("TypeScript", "tsx"),
    ".cpp": ("C++", "cpp"), ".c": ("C", "c"), ".h": ("C/C++ header", "c"), ".hpp": ("C/C++ header", "cpp"),
    ".java": ("Java", "java"), ".cs": ("C#", "csharp"), ".go": ("Go", "go"), ".rs": ("Rust", "rust"),
    ".rb": ("Ruby", "ruby"), ".php": ("PHP", "php"), ".kt": ("Kotlin", "kotlin"), ".swift": ("Swift", "swift"),
    ".sh": ("Shell", "bash"), ".sql": ("SQL", "sql"), ".md": ("Markdown", "markdown"),
    ".txt": ("Text", "text"), ".json": ("JSON", "json"), ".css": ("CSS", "css"), ".html": ("HTML", "html"),
    ".yaml": ("YAML", "yaml"), ".yml": ("YAML", "yaml"), ".toml": ("TOML", "toml"),
}
ALLOWED_EXTENSIONS = set(LANGUAGES)


def _lang(name: str):
    return getattr(Language, name, None)


_SPLITTER_LANG = {
    ".py": "PYTHON", ".js": "JS", ".jsx": "JS", ".ts": "TS", ".tsx": "TS",
    ".cpp": "CPP", ".c": "CPP", ".h": "CPP", ".hpp": "CPP", ".java": "JAVA",
    ".go": "GO", ".rs": "RUST", ".cs": "CSHARP", ".md": "MARKDOWN", ".html": "HTML",
    ".rb": "RUBY", ".php": "PHP", ".kt": "KOTLIN", ".swift": "SWIFT",
}
EXTENSION_TO_LANGUAGE = {e: _lang(n) for e, n in _SPLITTER_LANG.items() if _lang(n) is not None}


def language_label(path: str) -> str:
    return LANGUAGES.get(os.path.splitext(path)[1].lower(), ("Other", "text"))[0]


def code_language(path: str) -> str:
    return LANGUAGES.get(os.path.splitext(path)[1].lower(), ("Other", "text"))[1]


def _make_splitter(ext: str, size: int, overlap: int):
    kwargs = dict(chunk_size=size, chunk_overlap=overlap, add_start_index=True)
    if ext in EXTENSION_TO_LANGUAGE:
        return RecursiveCharacterTextSplitter.from_language(language=EXTENSION_TO_LANGUAGE[ext], **kwargs)
    return RecursiveCharacterTextSplitter(**kwargs)


def load_and_chunk_files(
    repo_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
    max_chunks: int = 6000,
    max_file_bytes: int = 300_000,
) -> Tuple[List[Dict[str, Any]], int]:
    """Walks a repo and returns (chunks, skipped_files).

    Each chunk carries source path, line range, extension and a content hash so
    the vector store can update incrementally. Shallow files are processed first,
    so if a repo exceeds `max_chunks` the deepest files are the ones skipped.
    """
    candidates = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if name in IGNORED_FILES or ext not in ALLOWED_EXTENSIONS:
                continue
            path = os.path.join(root, name)
            try:
                size = os.path.getsize(path)
            except OSError:
                continue
            if size == 0 or size > max_file_bytes:
                continue
            rel = os.path.relpath(path, repo_path).replace(os.sep, "/")
            candidates.append((rel.count("/"), rel, path, ext))
    candidates.sort()

    splitters: Dict[str, Any] = {}
    chunks: List[Dict[str, Any]] = []
    skipped = 0

    for _, rel, path, ext in candidates:
        try:
            with open(path, "rb") as f:
                data = f.read()
            content = data.decode("utf-8", errors="ignore")
            if not content.strip() or "\x00" in content:
                continue

            splitter = splitters.setdefault(ext, _make_splitter(ext, chunk_size, chunk_overlap))
            docs = splitter.create_documents([content])
            if len(chunks) + len(docs) > max_chunks:
                skipped += 1
                continue

            file_hash = hashlib.md5(data).hexdigest()
            for idx, d in enumerate(docs):
                start = max(d.metadata.get("start_index", 0), 0)
                start_line = content.count("\n", 0, start) + 1
                chunks.append({
                    "page_content": d.page_content,
                    "metadata": {
                        "source": rel,
                        "ext": ext,
                        "chunk_index": idx,
                        "total_chunks": len(docs),
                        "start_line": start_line,
                        "end_line": start_line + d.page_content.count("\n"),
                        "file_hash": file_hash,
                    },
                })
        except Exception as e:  # keep indexing even if one file is odd
            print(f"Skipping {path}: {e}")
            skipped += 1

    return chunks, skipped


def build_tree(sources: Iterable[str], root: str = "repo", max_lines: int = 300) -> str:
    """ASCII tree built from a list of relative file paths (no clone needed)."""
    tree: Dict[str, Any] = {}
    for s in sorted(sources):
        node = tree
        parts = s.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = None

    lines = [f"{root}/"]

    def render(node: Dict[str, Any], prefix: str = "") -> None:
        items = sorted(node.items(), key=lambda kv: (kv[1] is None, kv[0].lower()))
        for i, (name, child) in enumerate(items):
            last = i == len(items) - 1
            lines.append(f"{prefix}{'└── ' if last else '├── '}{name}{'/' if child is not None else ''}")
            if child is not None:
                render(child, prefix + ("    " if last else "│   "))

    render(tree)
    if len(lines) > max_lines:
        extra = len(lines) - max_lines
        lines = lines[:max_lines] + [f"... ({extra} more entries)"]
    return "\n".join(lines)
