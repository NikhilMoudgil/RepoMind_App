import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Backend module imports
from backend.query_decomposer import decompose_query
from backend.repo_loader import clone_repository
from backend.file_parser import load_and_chunk_files, generate_repo_tree
from backend.vector_store import index_chunks_in_vector_db, search_similar_code

load_dotenv()

st.set_page_config(page_title="RepoMind", layout="wide", page_icon="🧠")
st.title("🧠 RepoMind: Chat with your Codebase")

# Fetch API Key
api_key = None
try:
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    pass 

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    st.error("GROQ_API_KEY is missing. Please add it to your Streamlit secrets or local .env file.")
    st.stop()

client = Groq(api_key=api_key)

# Initialize Session State
if "repo_name" not in st.session_state:
    st.session_state.repo_name = None
if "repo_path" not in st.session_state:
    st.session_state.repo_path = None

# Sidebar Setup
st.sidebar.header("⚙️ Repository Setup")
repo_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")

if st.sidebar.button("Clone & Index Repository", use_container_width=True):
    if not repo_url:
        st.sidebar.warning("Please enter a repository URL.")
    else:
        with st.sidebar.status("Processing Repository...", expanded=True) as status:
            try:
                st.write("📥 Cloning repository...")
                local_path = clone_repository(repo_url)
                st.session_state.repo_path = local_path
                st.session_state.repo_name = os.path.basename(os.path.normpath(local_path))
                
                st.write("⚙️ Parsing and chunking files...")
                chunks = load_and_chunk_files(st.session_state.repo_path)
                
                if not chunks:
                    status.update(label="No valid code found", state="error", expanded=True)
                else:
                    st.write("🗄️ Indexing into Vector DB...")
                    indexed_count = index_chunks_in_vector_db(st.session_state.repo_name, chunks)
                    status.update(label=f"Ready! Indexed {indexed_count} chunks.", state="complete", expanded=False)
                    st.toast(f"Successfully loaded {st.session_state.repo_name}!", icon="🎉")
                    
            except Exception as e:
                status.update(label="Error processing repository", state="error", expanded=True)
                st.error(f"Details: {str(e)}")

# Main Query Interface
if st.session_state.repo_name:
    st.markdown(f"### 📂 Currently querying: **`{st.session_state.repo_name}`**")
    st.divider()
    
    prompt = st.chat_input("Ask a question about this codebase...")
    
    if prompt:
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing codebase architecture..."):
                try:
                    # STEP 1: Decompose query into targeted search queries
                    sub_queries = decompose_query(prompt, client)
                    
                    # STEP 2: Retrieve and deduplicate snippets across all sub-queries
                    retrieved_map = {}
                    for q in sub_queries:
                        snippets = search_similar_code(st.session_state.repo_name, q, top_k=2)
                        for s in snippets:
                            source = s["metadata"].get("source", "unknown")
                            chunk_idx = s["metadata"].get("chunk_index", 0)
                            unique_key = f"{source}_{chunk_idx}"
                            retrieved_map[unique_key] = s

                    relevant_snippets = list(retrieved_map.values())[:5]
                    
                    # Format context snippets string
                    context_str = ""
                    for snippet in relevant_snippets:
                        source = snippet["metadata"].get("source", "unknown")
                        context_str += f"\n--- File: {source} ---\n{snippet['page_content']}\n"

                    # STEP 3: Generate repository tree layout
                    repo_tree = generate_repo_tree(st.session_state.repo_path) if st.session_state.repo_path else ""

                    # STEP 4: Construct System & User Prompt with context + repo tree
                    system_prompt = (
                        "You are RepoMind, an expert AI assistant that analyzes codebases.\n"
                        "Below is the file tree structure of the repository:\n"
                        f"```\n{repo_tree}\n```\n\n"
                        "Use both the file tree structure and the provided code snippets to answer the user's prompt accurately.\n"
                        "If the answer cannot be found in the snippets, mention that clearly."
                    )
                    user_prompt = f"Code Context:\n{context_str}\n\nUser Question:\n{prompt}"

                    # STEP 5: Execute Groq LLM Completion
                    completion = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=1024
                    )
                    
                    st.write(completion.choices[0].message.content)

                    # Display search queries and retrieved context
                    with st.expander("🔍 View Search Queries & Retrieved Source Code"):
                        st.markdown("**Generated Sub-Queries:**")
                        for sq in sub_queries:
                            st.caption(f"• `{sq}`")
                        st.divider()
                        for snippet in relevant_snippets:
                            source = snippet["metadata"].get("source", "unknown")
                            st.markdown(f"**{source}**")
                            st.code(snippet['page_content'], language='python')
                
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")
else:
    st.info("👈 Please clone and index a repository using the sidebar to start chatting.")