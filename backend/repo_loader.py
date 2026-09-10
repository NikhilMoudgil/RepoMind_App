import os
import shutil
import stat
from git import Repo

REPOS_DIR = os.path.join(os.path.dirname(__file__), "cloned_repos")

def _remove_readonly(func, path, _):
    """Clear the read-only attribute on Windows files and retry deletion."""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_repository(repo_url: str) -> str:
    """Clones a public GitHub repository locally and returns the local directory path."""
    os.makedirs(REPOS_DIR, exist_ok=True)
    
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    target_path = os.path.join(REPOS_DIR, repo_name)
    
    # If the repository was previously cloned, force-remove read-only files
    if os.path.exists(target_path):
        shutil.rmtree(target_path, onerror=_remove_readonly)
        
    print(f"Cloning {repo_url} into {target_path}...")
    Repo.clone_from(repo_url, target_path)
    
    return target_path