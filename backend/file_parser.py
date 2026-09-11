import os
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
from langchain_text_splitters import Language # type: ignore
# Directory/file patterns and extensions to ignore
IGNORED_DIRS = {".git", "node_modules", "venv", "__pycache__", "dist", "build", ".vscode", ".idea"}
IGNORED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".pdf", ".zip", ".tar", ".gz", ".exe", ".bin", ".lock"}
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".cpp", ".c",
    ".h", ".hpp",  ".java", ".cs", ".go", ".rs", ".md", 
    ".txt", ".json", ".css", ".html", ".yaml", ".yml"}
EXTENSION_TO_LANGUAGE ={
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".jsx": Language.JS,
    ".ts": Language.TS,
    ".tsx": Language.TS,
    ".cpp": Language.CPP,
    ".c": Language.CPP,
    ".h": Language.CPP,
    ".hpp": Language.CPP,
    ".java": Language.JAVA,
    ".go": Language.GO,
    ".rs": Language.RUST,
    ".cs": Language.CSHARP,
    ".md": Language.MARKDOWN,
}
def load_and_chunk_files(repo_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Dict[str, Any]]:
    """Traverses a cloned repository, reads allowed code files, and chunks them into text blocks with metadata."""
    documents = []

    # Initialize code/text splitter
    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    for root, dirs, files in os.walk(repo_path):
        # Filter out ignored directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            
            # Process only text and code files
            if ext in ALLOWED_EXTENSIONS and ext not in IGNORED_EXTENSIONS:
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, repo_path)

                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    if not content.strip():
                        continue
                    # Select language-specific splitter or standard text splitter
                    if ext in EXTENSION_TO_LANGUAGE:
                        text_splitter = RecursiveCharacterTextSplitter.from_language(
                            language=EXTENSION_TO_LANGUAGE[ext],
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                    else:
                        text_splitter = default_splitter

                    # Split raw file content into chunks
                    chunks = text_splitter.split_text(content)

                    for idx, chunk in enumerate(chunks):
                        documents.append({
                            "page_content": chunk,  # Key matches vector_store expectations
                            "metadata": {
                                "source": relative_path,  # Key matches vector_store expectations
                                "chunk_index": idx,
                                "total_chunks": len(chunks)
                            }
                        })
                except Exception as e:
                    print(f"Skipping file {file_path} due to error: {e}")

    return documents
def generate_repo_tree(repo_path: str, max_depth: int = 4) -> str:
    """Generates an ASCII text tree representing the repository directory layout."""
    tree_lines = []

    for root, dirs, files in os.walk(repo_path):
        # Ignore hidden/build directories in-place
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

        rel_path = os.path.relpath(root, repo_path)
        depth = 0 if rel_path == "." else rel_path.count(os.sep) + 1

        if depth > max_depth:
            continue

        indent = "  " * depth
        folder_name = os.path.basename(root) if rel_path != "." else os.path.basename(repo_path)
        tree_lines.append(f"{indent}📁 {folder_name}/")

        sub_indent = "  " * (depth + 1)
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in ALLOWED_EXTENSIONS and ext not in IGNORED_EXTENSIONS:
                tree_lines.append(f"{sub_indent}📄 {f}")

    return "\n".join(tree_lines)