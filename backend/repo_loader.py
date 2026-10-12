import os
import re
import shutil
import stat
import tempfile
from typing import Optional, Tuple

from git import Repo

# Clones are temporary: once indexed, the vectors live in ChromaDB.
REPOS_DIR = os.path.join(tempfile.gettempdir(), "repomind_clones")

GITHUB_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?(?:/.*)?$"
)


def _remove_readonly(func, path, _):
    """Clear the read-only attribute (Windows .git files) and retry."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


def parse_github_url(url: str) -> Tuple[str, str]:
    m = GITHUB_RE.match(url.strip())
    if not m:
        raise ValueError("That doesn't look like a GitHub repository URL (https://github.com/owner/repo).")
    return m.group(1), m.group(2)


def clone_repository(repo_url: str, branch: Optional[str] = None) -> Tuple[str, str, str]:
    """Shallow-clones a public GitHub repo.

    Returns (local_path, repo_id, clean_url) where repo_id is "owner/name".
    """
    owner, name = parse_github_url(repo_url)
    clean_url = f"https://github.com/{owner}/{name}.git"
    repo_id = f"{owner}/{name}"

    os.makedirs(REPOS_DIR, exist_ok=True)
    target = os.path.join(REPOS_DIR, f"{owner}__{name}")
    if os.path.exists(target):
        shutil.rmtree(target, onerror=_remove_readonly)

    kwargs = {"depth": 1, "single_branch": True, "env": {"GIT_TERMINAL_PROMPT": "0"}}
    if branch:
        kwargs["branch"] = branch
    Repo.clone_from(clean_url, target, **kwargs)
    return target, repo_id, clean_url


def cleanup_clone(path: str) -> None:
    if path and os.path.exists(path):
        shutil.rmtree(path, onerror=_remove_readonly)
