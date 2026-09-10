import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq  # type: ignore

from backend.repo_loader import clone_repository
from backend.file_parser import load_and_chunk_files
from backend.vector_store import index_chunks_in_vector_db, search_similar_code

load_dotenv()

app = FastAPI(title="RepoMind Backend")

api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

class RepoCloneRequest(BaseModel):
    repo_url: str

class RepoIndexRequest(BaseModel):
    repo_path: str

class QueryRepoRequest(BaseModel):
    repo_name: str
    prompt: str

@app.get("/")
def home():
    return {"status": "RepoMind backend is running!"}

@app.post("/clone-repo")
def clone_repo_endpoint(data: RepoCloneRequest):
    try:
        local_path = clone_repository(data.repo_url)
        return {
            "status": "success",
            "message": "Repository cloned successfully.",
            "path": local_path
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/index-repo")
def index_repo_endpoint(data: RepoIndexRequest):
    if not os.path.exists(data.repo_path):
        raise HTTPException(status_code=404, detail="Directory does not exist.")
    
    try:
        # Step 1: Parse and chunk repository using load_and_chunk_files
        chunks = load_and_chunk_files(data.repo_path)
        if not chunks:
            return {"status": "warning", "message": "No valid code files found to index."}

        # Step 2: Index in Vector DB
        repo_name = os.path.basename(os.path.normpath(data.repo_path))
        indexed_count = index_chunks_in_vector_db(repo_name, chunks)

        return {
            "status": "success",
            "repo_name": repo_name,
            "total_chunks_indexed": indexed_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query-repo")
def query_repo_endpoint(data: QueryRepoRequest):
    if not api_key or not client:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is missing from environment.")

    # Step 1: Retrieve context snippets from Vector DB
    relevant_snippets = search_similar_code(data.repo_name, data.prompt, top_k=2)
    
    context_str = ""
    for snippet in relevant_snippets:
        source = snippet["metadata"].get("source", "unknown")
        context_str += f"\n--- File: {source} ---\n{snippet['page_content']}\n"

    # Step 2: Construct RAG prompt
    system_prompt = (
        "You are RepoMind, an expert AI assistant that analyzes codebases.\n"
        "Use the provided code snippets from the repository to answer the user's prompt accurately.\n"
        "If the answer cannot be found in the snippets, mention that clearly."
    )
    
    user_prompt = f"Code Context:\n{context_str}\n\nUser Question:\n{data.prompt}"

    # Step 3: Generate response via Groq using an active, supported model
    try:
        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",  # Updated from deprecated llama3-70b-8192
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1024
        )
    except Exception as e:
        print(f"GROQ API EXCEPTION DETAIL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")
    
    return {
        "response": completion.choices[0].message.content,
        "retrieved_context": relevant_snippets
    }