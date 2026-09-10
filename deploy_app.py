import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

# Import your existing backend logic
from backend.repo_loader import clone_repository
from backend.file_parser import load_and_chunk_files
from backend.vector_store import index_chunks_in_vector_db, search_similar_code

# Load environment variables from local .env file (ignored in cloud)
load_dotenv()

# Configure page settings
st.set_page_config(page_title="RepoMind", layout="wide", page_icon="🧠")

st.title("🧠 RepoMind: Chat with your Codebase")

# Safely fetch API key: Try Streamlit Secrets first (Cloud), fallback to os.getenv (Local)
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

# Sleek Sidebar UI
st.sidebar.header("⚙️ Repository Setup")
repo_url = st.sidebar.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")

if st.sidebar.button("Clone & Index Repository", use_container_width=True):
    if not repo_url:
        st.sidebar.warning("Please enter a repository URL.")
    else:
        # Use st.status for a clean, collapsible loading sequence
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
                    
                    # Update status to complete and collapse the box
                    status.update(label=f"Ready! Indexed {indexed_count} chunks.", state="complete", expanded=False)
                    
                    # Ephemeral popup notification
                    st.toast(f"Successfully loaded {st.session_state.repo_name}!", icon="🎉")
                    
            except Exception as e:
                status.update(label="Error processing repository", state="error", expanded=True)
                st.error(f"Details: {str(e)}")

# Main Query Interface
if st.session_state.repo_name:
    st.markdown(f"### 📂 Currently querying: **`{st.session_state.repo_name}`**")
    st.divider()
    
    # Modern Chat Input instead of a clunky text area
    prompt = st.chat_input("Ask a question about this codebase...")
    
    if prompt:
        # Display user message bubble
        with st.chat_message("user"):
            st.write(prompt)
            
        # Display assistant message bubble
        with st.chat_message("assistant"):
            with st.spinner("Analyzing codebase..."):
                try:
                    # Retrieve context snippets
                    relevant_snippets = search_similar_code(st.session_state.repo_name, prompt, top_k=3)
                    
                    context_str = ""
                    for snippet in relevant_snippets:
                        source = snippet["metadata"].get("source", "unknown")
                        context_str += f"\n--- File: {source} ---\n{snippet['page_content']}\n"

                    # Construct RAG prompt
                    system_prompt = (
                        "You are RepoMind, an expert AI assistant that analyzes codebases.\n"
                        "Use the provided code snippets from the repository to answer the user's prompt accurately.\n"
                        "If the answer cannot be found in the snippets, mention that clearly."
                    )
                    user_prompt = f"Code Context:\n{context_str}\n\nUser Question:\n{prompt}"

                    # Generate response via Groq
                    completion = client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_tokens=1024
                    )
                    
                    st.write(completion.choices[0].message.content)

                    # Collapsible expander for raw context
                    with st.expander("🔍 View Retrieved Source Code"):
                        for snippet in relevant_snippets:
                            source = snippet["metadata"].get("source", "unknown")
                            st.markdown(f"**{source}**")
                            st.code(snippet['page_content'], language='python')
                
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")
else:
    # Empty state UI
    st.info("👈 Please clone and index a repository using the sidebar to start chatting.")