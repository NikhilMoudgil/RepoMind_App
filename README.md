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
# 🚀 Getting Started

## Prerequisites

Before running RepoMind, make sure you have the following installed:

- **Python 3.10+**
- **Git**
- A free **Groq API Key** from [Groq Console](https://console.groq.com/)

---

## 1. Clone RepoMind & Set Up Virtual Environment

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RepoMind.git
cd RepoMind

Create a virtual environment
python -m venv venv
Activate the environment

Windows (PowerShell):

.\venv\Scripts\Activate.ps1

Mac/Linux:

source venv/bin/activate
2. Install Dependencies

Install the backend dependencies:

pip install -r backend/requirements.txt

Install the frontend dependencies:

pip install -r frontend/requirements.txt
3. Configure API Key

Create a .env file in the root RepoMind/ directory:

GROQ_API_KEY=your_groq_api_key_here

Replace your_groq_api_key_here with your actual Groq API key.

Important: Never commit your .env file or API key to GitHub.

4. Run the Project

RepoMind consists of a FastAPI backend and a Streamlit frontend.

Run both components in separate terminal windows.

Terminal 1 — Start Backend API

From the project root:

cd backend
uvicorn main:app --reload

The backend API will run at:

http://127.0.0.1:8000
Terminal 2 — Start Frontend UI

From the root directory with the virtual environment activated:

streamlit run frontend/app.py

The Streamlit dashboard will be available at:

http://localhost:8501
📖 How to Use

Open the Streamlit dashboard in your browser:

http://localhost:8501

Enter any public GitHub repository URL in the sidebar.

Example:

https://github.com/user/repository
Click Clone & Index Repository.
Wait for the repository indexing process to complete.
Once indexing is complete, ask questions about the codebase using the chat interface.
Example Questions
What data structures or algorithms are implemented in this repository?
Explain how the main application entry point works.
Show me where exception handling is implemented.
🔧 Common Troubleshooting
WinError 10061 — Connection Refused

Problem:
The Streamlit frontend cannot connect to the FastAPI backend.

Solution:

Make sure the backend server is running in another terminal:

cd backend
uvicorn main:app --reload

Then restart the Streamlit frontend if necessary:

streamlit run frontend/app.py
WinError 5 — Access Denied

Problem:
Git may mark .git files as read-only on Windows, preventing the existing cloned repository from being deleted.

Solution:

Make sure repo_loader.py contains a chmod error handler when deleting existing repository clones.

This allows read-only Git files to have their permissions changed before deletion.


🛡️ Security

Never expose your Groq API key publicly.

Make sure .env is included in .gitignore:

.env
venv/
__pycache__/
🚀 RepoMind

RepoMind allows developers to clone, index, and interact with GitHub repositories using natural language.

Instead of manually searching through a large codebase, developers can ask questions and get contextual answers about the repository.
