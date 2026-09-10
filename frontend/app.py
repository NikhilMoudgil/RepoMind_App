import os
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RepoMind - AI Codebase Assistant", page_icon="🧠", layout="wide")

st.title("🧠 RepoMind: Talk to Your Codebase")
st.caption("Clone, index, and query any public GitHub repository using RAG.")

if "active_repo" not in st.session_state:
    st.session_state.active_repo = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("📂 Repository Setup")
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/user/repo")
    
    if st.button("Clone & Index Repository"):
        if not repo_url:
            st.error("Please enter a valid GitHub repository URL.")
        else:
            with st.spinner("Cloning repository..."):
                try:
                    # Step 1: Clone repo
                    clone_res = requests.post(f"{BACKEND_URL}/clone-repo", json={"repo_url": repo_url})
                    if clone_res.status_code != 200:
                        st.error(f"Clone failed: {clone_res.json().get('detail')}")
                    else:
                        local_path = clone_res.json().get("path")
                        
                        # Step 2: Index repo
                        with st.spinner("Indexing code chunks into vector database..."):
                            index_res = requests.post(f"{BACKEND_URL}/index-repo", json={"repo_path": local_path})
                            
                            if index_res.status_code == 200:
                                data = index_res.json()
                                if data.get("status") == "warning":
                                    st.warning(data.get("message", "No valid code files found to index."))
                                else:
                                    st.session_state.active_repo = data.get("repo_name")
                                    st.success(f"Successfully indexed '{st.session_state.active_repo}' ({data.get('total_chunks_indexed')} chunks)!")
                            else:
                                st.error(f"Indexing failed: {index_res.json().get('detail')}")
                except Exception as e:
                    st.error(f"Backend connection error: {str(e)}")

    if st.session_state.active_repo:
        st.info(f"Active Repository: **{st.session_state.active_repo}**")

# Main Chat Interface
if not st.session_state.active_repo:
    st.warning("Please clone and index a repository using the sidebar to start asking questions.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_query := st.chat_input(f"Ask a question about {st.session_state.active_repo}..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching codebase..."):
                try:
                    payload = {
                        "repo_name": st.session_state.active_repo,
                        "prompt": user_query
                    }
                    response = requests.post(f"{BACKEND_URL}/query-repo", json=payload)
                    
                    if response.status_code == 200:
                        ai_reply = response.json().get("response")
                        st.markdown(ai_reply)
                        st.session_state.messages.append({"role": "assistant", "content": ai_reply})
                    else:
                        st.error("Failed to generate response from backend.")
                except Exception as e:
                    st.error(f"Error communicating with backend: {str(e)}")