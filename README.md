# 🧠 RepoMind: Talk to Your Codebase

RepoMind is an intelligent Retrieval-Augmented Generation (RAG) platform that enables developers to clone, index, and query any public GitHub repository using natural language. Built with FastAPI, Streamlit, ChromaDB, and Groq LLMs.

## ✨ Features
* **📂 Automated Repository Cloning:** Fetch any public GitHub repository on demand.
* **🔍 Smart Code Chunking & Parsing:** Support for .py, .js, .ts, .cpp, .c, .java, .md, and more.
* **⚡ High-Speed Vector Search:** Embedded with sentence-transformers and indexed locally in ChromaDB.
* **💬 AI-Powered Code Analysis:** Context-aware Q&A leveraging Groq's high-performance LLM API.
* **📊 Structured UI:** Clean Streamlit dashboard with expandable context and formatted Markdown responses.

## 🛠️ Tech Stack
* **Frontend UI:** Streamlit
* **Backend Framework:** FastAPI + Uvicorn
* **Vector Database:** ChromaDB
* **Embeddings:** sentence-transformers
* **LLM Provider:** Groq API
* **Repository Management:** GitPython

## 📁 Project Structure

```text
RepoMind/
├── backend/
│   ├── main.py              # FastAPI endpoints (/clone-repo, /index-repo, /query-repo)
│   ├── repo_loader.py       # Git cloning and directory cleanup
│   ├── file_parser.py       # File extension filtering and text chunking
│   ├── vector_store.py      # ChromaDB setup and embedding storage
│   └── cloned_repos/        # Local workspace for cloned target repositories
├── frontend/
│   ├── app.py               # Streamlit chat interface and API calls
│   └── requirements.txt     # Frontend-specific dependencies
├── .env                     # Environment variables (API keys)
├── .gitignore
└── README.md


🚀 Getting StartedPrerequisitesPython 3.10+  Git installed on your operating system  A free Groq API Key (obtainable at console.groq.com)  1. Clone RepoMind & Set Up Virtual EnvironmentBashgit clone [https://github.com/YOUR_USERNAME/RepoMind.git](https://github.com/YOUR_USERNAME/RepoMind.git)
cd RepoMind

# Create virtual environment
python -m venv venv
Activate the environment:Windows (PowerShell): .\venv\Scripts\Activate.ps1  Mac/Linux: source venv/bin/activate  2. Install DependenciesBashpip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
