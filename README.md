🧠 RepoMind: Talk to Your CodebaseRepoMind is an intelligent Retrieval-Augmented Generation (RAG) platform that enables developers to clone, index, and query any public GitHub repository using natural language. Built with FastAPI, Streamlit, ChromaDB, and Groq LLMs.✨ Features📂 Automated Repository Cloning: Fetch any public GitHub repository on demand.🔍 Smart Code Chunking & Parsing: Support for .py, .js, .ts, .cpp, .c, .java, .md, and more.⚡ High-Speed Vector Search: Embedded with sentence-transformers and indexed locally in ChromaDB.💬 AI-Powered Code Analysis: Context-aware Q&A leveraging Groq's high-performance LLM API.📊 Structured UI: Clean Streamlit dashboard with expandable context and formatted Markdown responses.🛠️ Tech StackLayerTechnologyFrontend UIStreamlitBackend FrameworkFastAPI + UvicornVector DatabaseChromaDBEmbeddingssentence-transformersLLM ProviderGroq APIRepository ManagementGitPython📁 Project StructurePlaintextRepoMind/
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
🚀 Getting StartedPrerequisitesPython 3.10+Git installed on your operating systemA free Groq API Key (obtainable at console.groq.com)1. Clone RepoMind & Set Up Virtual EnvironmentBashgit clone https://github.com/YOUR_USERNAME/RepoMind.git
cd RepoMind

# Create virtual environment
python -m venv venv
Activate the environment:Windows (PowerShell):PowerShell.\venv\Scripts\Activate.ps1
Mac/Linux:Bashsource venv/bin/activate
2. Install DependenciesBashpip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
3. Configure API KeyCreate a .env file in the root RepoMind/ directory:Code snippetGROQ_API_KEY=your_groq_api_key_here
4. Run the ProjectLaunch both the FastAPI backend and Streamlit frontend in separate terminal windows.Terminal 1: Start Backend APIBashcd backend
uvicorn main:app --reload
The API will run locally at [http://127.0.0.1:8000](http://127.0.0.1:8000).Terminal 2: Start Frontend UIBash# From the root directory with venv activated
streamlit run frontend/app.py
The web dashboard will open at http://localhost:8501.📖 How to UseOpen http://localhost:8501 in your web browser.Enter any public GitHub repository URL into the sidebar (e.g., [https://github.com/user/repository](https://github.com/user/repository)).Click Clone & Index Repository.Once indexing completes, ask questions about the codebase in the chat bar:"What data structures or algorithms are implemented in this repository?""Explain how the main application entry point works.""Show me where exception handling is implemented."🔧 Common TroubleshootingWinError 10061 (Connection Refused): The Streamlit frontend cannot reach the backend. Verify that Uvicorn is active in Terminal 1.WinError 5 (Access Denied): Git marks .git files as read-only on Windows. Ensure repo_loader.py includes a chmod error handler when deleting existing clones.