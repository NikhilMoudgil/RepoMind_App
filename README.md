---
title: RepoMind
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# RepoMind

Chat with any public GitHub repository. RepoMind clones it, splits the code into chunks, embeds them in ChromaDB, and answers questions with a Groq-hosted model, citing the file and line range it used.

## What's in it
- **Chat** with streaming answers, follow-up memory, and source snippets with line numbers
- **File map** with an ASCII tree, language breakdown and a filterable file table
- **Briefing**: one-click architecture summary from the README, config files and entry points
- Incremental re-indexing (unchanged files are skipped, deleted files are purged)
- Naming a file in a question ("explain repo_loader.py") pulls that file in directly
- Multiple indexed repos, model picker, and chat export

## Run locally
```bash
python -m venv venv && source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
echo "GROQ_API_KEY=your_key" > .env
streamlit run app.py
```

## Deploy free on Hugging Face Spaces
1. Create a Space at huggingface.co/new-space, choose **Streamlit** as the SDK (free CPU, 16 GB RAM).
2. Upload everything in this folder, with `app.py` at the root. The block at the top of this README is the Space config.
3. Space **Settings → Variables and secrets → New secret**: `GROQ_API_KEY` = your key. (If you skip this, visitors paste their own key in the sidebar.)
4. The Space builds and starts. The first question downloads a ~80 MB embedding model once.

Notes: free Space storage is ephemeral, so the index is rebuilt after a restart or sleep. The LLM runs on Groq, so the Space only does embeddings and search.

## Rate limits
When visitors use the server's `GROQ_API_KEY`, each visitor is limited per time window, and all visitors together are capped, so one person can't use up your free Groq quota. Visitors who paste their own key in the sidebar are not limited. If Groq itself returns a rate-limit error, the app shows a short "try again" message instead of a raw error.

Tune it with environment variables (Space **Variables**, or `.env` locally). Use `0` to turn a cap off:

| Variable | Default | Meaning |
|---|---|---|
| `RATE_LIMIT_PER_SESSION` | `10` | questions per visitor per window |
| `RATE_LIMIT_GLOBAL` | `60` | questions across all visitors per window |
| `RATE_LIMIT_WINDOW_MIN` | `10` | window length in minutes |
