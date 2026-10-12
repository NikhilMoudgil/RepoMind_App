import re
from typing import Any, Dict, Iterator, List

from groq import Groq

from . import vector_store as vs

MODELS = {
    "GPT-OSS 120B · best quality": "openai/gpt-oss-120b",
    "GPT-OSS 20B · faster": "openai/gpt-oss-20b",
    "Llama 3.3 70B · balanced": "llama-3.3-70b-versatile",
    "Llama 3.1 8B · fastest": "llama-3.1-8b-instant",
}

SYSTEM_PROMPT = """You are RepoMind, a senior engineer who explains codebases.

You are given the repository's file tree and a set of code excerpts retrieved for the user's question.
Each excerpt is labelled like [1] path/to/file.py:L10-L42.

Rules:
- Ground every claim in the excerpts or the file tree. If the excerpts don't contain the answer, say what is missing and which files are likely to hold it.
- Cite sources inline using the label format, e.g. `backend/main.py:L10-L42`.
- Prefer concrete explanations (what a function does, how data flows) over generic descriptions.
- Use Markdown. Put code in fenced blocks with a language tag. Keep answers focused."""

MANIFEST_NAMES = (
    "readme.md", "readme.txt", "package.json", "pyproject.toml", "requirements.txt", "setup.py",
    "cargo.toml", "go.mod", "pom.xml", "dockerfile", "docker-compose.yml", "main.py", "app.py",
    "index.js", "index.ts", "main.go", "main.rs",
)


def _is_reasoning(model: str) -> bool:
    return model.startswith("openai/gpt-oss")


def find_mentioned_files(question: str, sources: List[str], limit: int = 3) -> List[str]:
    """If the question names a file ('explain main.py'), return its full path in the repo."""
    found: List[str] = []
    for token in re.findall(r"[\w./-]+\.\w+", question):
        for s in sources:
            if (s == token or s.endswith("/" + token)) and s not in found:
                found.append(s)
    return found[:limit]


def retrieve(
    repo_id: str,
    queries: List[str],
    sources: List[str],
    question: str,
    top_k: int = 4,
    max_snippets: int = 8,
) -> List[Dict[str, Any]]:
    """Merges named-file chunks and vector hits from every sub-query, best match first."""
    pool: Dict[str, Dict[str, Any]] = {}

    def add(snippet: Dict[str, Any]) -> None:
        m = snippet["metadata"]
        key = f"{m['source']}::{m.get('chunk_index', 0)}"
        if key not in pool or snippet["distance"] < pool[key]["distance"]:
            pool[key] = snippet

    for src in find_mentioned_files(question, sources):
        for s in vs.get_file_chunks(repo_id, src, limit=2):
            add(s)
    for q in queries:
        for s in vs.search(repo_id, q, top_k=top_k):
            add(s)

    return sorted(pool.values(), key=lambda s: s["distance"])[:max_snippets]


def format_context(snippets: List[Dict[str, Any]]) -> str:
    parts = []
    for i, s in enumerate(snippets, 1):
        m = s["metadata"]
        label = f"{m['source']}:L{m.get('start_line', '?')}-L{m.get('end_line', '?')}"
        parts.append(f"[{i}] {label}\n```\n{s['page_content']}\n```")
    return "\n\n".join(parts) if parts else "(no excerpts retrieved)"


def build_messages(tree: str, snippets, question: str, history: List[dict]) -> List[dict]:
    system = f"{SYSTEM_PROMPT}\n\nFile tree:\n```\n{tree[:6000]}\n```"
    user = f"Retrieved excerpts:\n\n{format_context(snippets)}\n\nQuestion: {question}"
    return [{"role": "system", "content": system}, *history[-6:], {"role": "user", "content": user}]


def stream_chat(client: Groq, model: str, messages: List[dict], temperature: float = 0.2) -> Iterator[str]:
    kwargs: Dict[str, Any] = dict(
        model=model, messages=messages, temperature=temperature, max_tokens=4096, stream=True
    )
    if _is_reasoning(model):
        kwargs["reasoning_effort"] = "low"
    for chunk in client.chat.completions.create(**kwargs):
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def overview_messages(repo_id: str, tree: str, sources: List[str]) -> List[dict]:
    """Builds the prompt for the one-click architecture summary."""
    picked: List[str] = []
    for name in MANIFEST_NAMES:
        for s in sources:
            if s.lower().split("/")[-1] == name and s.count("/") <= 2 and s not in picked:
                picked.append(s)
    material = []
    for src in picked[:7]:
        for c in vs.get_file_chunks(repo_id, src, limit=2):
            material.append(f"### {src}\n```\n{c['page_content']}\n```")

    user = (
        f"Repository: {repo_id}\n\nFile tree:\n```\n{tree[:6000]}\n```\n\n"
        f"Key files:\n\n{chr(10).join(material) or '(none found)'}\n\n"
        "Write a briefing for a developer new to this project with these sections: "
        "**What it does**, **Tech stack**, **How it's organised** (the main modules and what each owns), "
        "**How data flows** (a short walk-through of the main path), **How to run it**, "
        "and **Good places to start reading**. Only state what the material supports; say so when something is unclear."
    )
    return [
        {"role": "system", "content": "You are a senior engineer writing a concise, accurate project briefing in Markdown."},
        {"role": "user", "content": user},
    ]
