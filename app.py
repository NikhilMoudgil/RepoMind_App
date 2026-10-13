# Some hosts ship an old SQLite that ChromaDB rejects; use pysqlite3 if it's installed.
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import os
import uuid
from collections import Counter

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq, RateLimitError

from backend import rag, vector_store as vs
from backend.file_parser import build_tree, code_language, language_label
from backend.query_decomposer import decompose_query
from backend.ratelimit import RateLimiter
from backend.repo_loader import cleanup_clone, clone_repository

load_dotenv()
MAX_CHUNKS = int(os.getenv("MAX_CHUNKS", "6000"))
# Limits apply only when visitors use YOUR server key. 0 disables a cap.
LIMIT_PER_SESSION = int(os.getenv("RATE_LIMIT_PER_SESSION", "10"))   # questions per visitor
LIMIT_GLOBAL = int(os.getenv("RATE_LIMIT_GLOBAL", "60"))             # questions across all visitors
LIMIT_WINDOW_MIN = int(os.getenv("RATE_LIMIT_WINDOW_MIN", "10"))     # window length in minutes

st.set_page_config(page_title="RepoMind", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- styling
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
.stApp { font-family: 'IBM Plex Sans', sans-serif; }
code, pre, .rm-mono { font-family: 'IBM Plex Mono', monospace !important; }
/* keep Streamlit's icon font intact, otherwise icons render as raw text */
span[data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {
    font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}
#MainMenu, footer, .stAppDeployButton { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }

section[data-testid="stSidebar"] { border-right: 1px solid #252D3A; }
section[data-testid="stSidebar"] .stButton > button { width: 100%; }

.rm-title { font-size: 1.7rem; font-weight: 600; letter-spacing: -0.02em; margin: 0; }
.rm-title a { color: inherit; text-decoration: none; }
.rm-title a:hover { text-decoration: underline; text-decoration-color: #7AA2F7; }
.rm-sub { color: #8A94A6; margin: .15rem 0 1.1rem; font-size: .95rem; }

/* one ledger strip instead of a row of identical cards */
.rm-ledger { display: flex; border: 1px solid #252D3A; border-radius: 6px; margin: 0 0 1.4rem; overflow: hidden; }
.rm-ledger > div { flex: 1; padding: .7rem 1rem; border-right: 1px solid #252D3A; }
.rm-ledger > div:last-child { border-right: none; }
.rm-ledger b { display: block; font-size: 1.35rem; font-weight: 500; }
.rm-ledger span { color: #8A94A6; font-size: .8rem; }

.rm-hero { font-size: 2.6rem; line-height: 1.1; font-weight: 600; letter-spacing: -0.03em; margin: 1.5rem 0 .6rem; }
.rm-lede { color: #9AA4B5; font-size: 1.05rem; max-width: 40rem; margin-bottom: 1.6rem; }
.rm-steps { color: #9AA4B5; font-size: .95rem; line-height: 1.8; max-width: 40rem; }
.rm-steps code { color: #E0AF68; background: transparent; }

.rm-cite { font-family: 'IBM Plex Mono', monospace; font-size: .8rem; color: #E0AF68; margin: .9rem 0 .2rem; }
.rm-query { font-family: 'IBM Plex Mono', monospace; font-size: .78rem; color: #8A94A6; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state
ss = st.session_state
ss.setdefault("chats", {})       # repo_id -> list of messages
ss.setdefault("summaries", {})   # repo_id -> AI briefing
ss.setdefault("flash", None)
ss.setdefault("session_id", uuid.uuid4().hex)

EXAMPLE_REPOS = [
    "https://github.com/pallets/flask",
    "https://github.com/tiangolo/fastapi",
    "https://github.com/psf/requests",
]


def server_key() -> str | None:
    try:
        key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        key = None
    return key or os.getenv("GROQ_API_KEY")


def set_url(url: str) -> None:
    ss["repo_url"] = url


def ask(question: str) -> None:
    ss["pending_prompt"] = question


@st.cache_resource
def get_limiter() -> RateLimiter:
    return RateLimiter(LIMIT_PER_SESSION, LIMIT_GLOBAL, LIMIT_WINDOW_MIN * 60)


def allow_request() -> bool:
    """Counts one model request against the limits; shows a message and returns False if over."""
    if not using_server_key:
        return True
    ok, wait, scope = get_limiter().check(ss["session_id"])
    if ok:
        return True
    mins, secs = divmod(wait, 60)
    when = f"{mins} min {secs} s" if mins else f"{secs} s"
    if scope == "session":
        st.warning(f"You've reached the limit of {LIMIT_PER_SESSION} questions per {LIMIT_WINDOW_MIN} minutes. Try again in {when}, or add your own Groq key in the sidebar to remove the limit.")
    else:
        st.warning(f"The demo is busy right now. Try again in {when}, or add your own Groq key in the sidebar to skip the queue.")
    return False


def explain_error(e: Exception) -> str:
    if isinstance(e, RateLimitError):
        retry = None
        try:
            retry = e.response.headers.get("retry-after")
        except Exception:
            pass
        wait = f" Try again in about {retry} seconds." if retry else " Try again in a minute."
        return "Groq's rate limit was reached." + wait + " A smaller model in Answer settings uses fewer tokens."
    return f"The model request failed: {e}"


@st.cache_data(show_spinner=False)
def cached_files(repo_id: str, version: int):
    return vs.get_repo_files(repo_id)


# ---------------------------------------------------------------- indexing
def index_repository(url: str, branch: str) -> None:
    with st.status("Indexing repository", expanded=True) as status:
        path = None
        try:
            st.write("Cloning (shallow)…")
            path, repo_id, clean_url = clone_repository(url, branch or None)

            st.write("Splitting files into chunks…")
            from backend.file_parser import load_and_chunk_files
            chunks, skipped = load_and_chunk_files(path, max_chunks=MAX_CHUNKS)
            if not chunks:
                status.update(label="No supported source files found", state="error")
                return

            st.write(f"Embedding {len(chunks):,} chunks…")
            bar = st.progress(0.0)
            stats = vs.index_chunks(repo_id, clean_url, chunks, progress=lambda d, t: bar.progress(d / max(t, 1)))

            msg = f"Indexed {repo_id}: {stats['added']} new, {stats['updated']} changed, {stats['unchanged']} unchanged files"
            if stats["removed"]:
                msg += f", {stats['removed']} removed"
            if skipped:
                msg += f". {skipped} large or deep files were skipped to stay under {MAX_CHUNKS:,} chunks"
            ss["flash"] = msg + "."
            ss["repo_select"] = repo_id
            ss["chats"].pop(repo_id, None)
            ss["summaries"].pop(repo_id, None)
            status.update(label="Ready", state="complete", expanded=False)
        except ValueError as e:
            status.update(label="Invalid URL", state="error")
            st.error(str(e))
        except Exception as e:
            status.update(label="Indexing failed", state="error")
            st.error(f"{e}\n\nCheck that the repository is public and the branch name is correct.")
        finally:
            if path:
                cleanup_clone(path)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## 🧠 RepoMind")
    st.caption("Ask questions about any public GitHub repository.")

    key = server_key()
    using_server_key = bool(key)
    if not key:
        key = st.text_input("Groq API key", type="password", help="Free at console.groq.com. Kept only in this browser session.")
    else:
        with st.expander("Use my own Groq key"):
            own = st.text_input("Groq API key", type="password", label_visibility="collapsed")
            if own:
                key = own
                using_server_key = False

    st.divider()
    st.markdown("**Add a repository**")
    url = st.text_input("GitHub URL", key="repo_url", placeholder="https://github.com/owner/repo", label_visibility="collapsed")
    branch = st.text_input("Branch", placeholder="Branch (optional, default branch if empty)", label_visibility="collapsed")
    if st.button("Clone and index", type="primary"):
        if not url.strip():
            st.warning("Paste a GitHub repository URL first.")
        else:
            index_repository(url, branch.strip())

    repos = vs.list_repos()
    if repos:
        st.divider()
        st.markdown("**Indexed repositories**")
        ids = [r["repo_id"] for r in repos]
        if ss.get("repo_select") not in ids:
            ss["repo_select"] = ids[0]
        st.selectbox("Active repository", ids, key="repo_select", label_visibility="collapsed")
        if st.button("Remove from index"):
            vs.delete_repo(ss["repo_select"])
            ss["chats"].pop(ss["repo_select"], None)
            ss["summaries"].pop(ss["repo_select"], None)
            ss.pop("repo_select", None)
            st.rerun()

    with st.expander("Answer settings"):
        model_label = st.selectbox("Model", list(rag.MODELS), index=0)
        top_k = st.slider("Snippets per search", 2, 8, 4)
        max_snippets = st.slider("Snippets sent to the model", 3, 12, 8)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
        show_sources = st.toggle("Show sources under answers", value=True)
    model = rag.MODELS[model_label]

client = Groq(api_key=key) if key else None

# ---------------------------------------------------------------- landing
if ss.get("flash"):
    st.success(ss.pop("flash"))

if not repos:
    st.markdown('<div class="rm-hero">Read a codebase by asking it questions.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rm-lede">Paste a GitHub link. RepoMind splits the code into searchable pieces, '
        "finds the ones that matter for each question, and answers with the file and line numbers it used.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("**Try one of these**")
    cols = st.columns(len(EXAMPLE_REPOS))
    for col, u in zip(cols, EXAMPLE_REPOS):
        col.button(u.replace("https://github.com/", ""), key=f"ex_{u}", on_click=set_url, args=(u,))
    st.markdown(
        '<div class="rm-steps" style="margin-top:1.4rem">'
        "Pick a repository above or paste your own in the sidebar, then choose <code>Clone and index</code>. "
        "Small repositories take under a minute. After that you can chat, browse the file map, "
        "or generate a project briefing."
        "</div>",
        unsafe_allow_html=True,
    )
    if not key:
        st.info("Add your Groq API key in the sidebar before asking questions.")
    st.stop()

# ---------------------------------------------------------------- repo header
repo_id = ss["repo_select"]
meta = next(r for r in repos if r["repo_id"] == repo_id)
files = cached_files(repo_id, meta["chunks"])
sources = [f["source"] for f in files]
lang_counts = Counter(language_label(f["source"]) for f in files)
top_lang = lang_counts.most_common(1)[0][0] if lang_counts else "–"
total_lines = sum(f["lines"] for f in files)

link = meta["repo_url"].removesuffix(".git") if meta["repo_url"] else f"https://github.com/{repo_id}"
st.markdown(f'<h1 class="rm-title"><a href="{link}" target="_blank">{repo_id}</a></h1>', unsafe_allow_html=True)
st.markdown('<div class="rm-sub">Indexed and ready for questions</div>', unsafe_allow_html=True)
st.markdown(
    f"""<div class="rm-ledger">
<div><b>{len(files):,}</b><span>files</span></div>
<div><b>{total_lines:,}</b><span>lines of code and docs</span></div>
<div><b>{meta['chunks']:,}</b><span>searchable chunks</span></div>
<div><b>{top_lang}</b><span>main language</span></div>
</div>""",
    unsafe_allow_html=True,
)

view = st.segmented_control("View", ["💬 Chat", "🗂️ File map", "✨ Briefing"], default="💬 Chat", label_visibility="collapsed") or "💬 Chat"
tree = build_tree(sources, root=repo_id.split("/")[-1])


# ---------------------------------------------------------------- helpers
def render_sources(snippets: list, queries: list) -> None:
    if not snippets:
        return
    with st.expander(f"Sources ({len(snippets)})"):
        if queries:
            st.markdown("<div class='rm-query'>Searched for: " + " &nbsp;|&nbsp; ".join(queries) + "</div>", unsafe_allow_html=True)
        for s in snippets:
            m = s["metadata"]
            st.markdown(f"<div class='rm-cite'>{m['source']}:L{m.get('start_line')}-L{m.get('end_line')}</div>", unsafe_allow_html=True)
            st.code(s["page_content"], language=code_language(m["source"]))


def chat_markdown(messages: list) -> str:
    return f"# RepoMind chat: {repo_id}\n\n" + "\n\n".join(
        f"**{'You' if m['role'] == 'user' else 'RepoMind'}:** {m['content']}" for m in messages
    )


# ---------------------------------------------------------------- chat
if view == "💬 Chat":
    messages = ss["chats"].setdefault(repo_id, [])
    prompt = st.chat_input(f"Ask about {repo_id.split('/')[-1]}…") or ss.pop("pending_prompt", None)

    if not messages and not prompt:
        st.markdown("**Start with a question**")
        starters = [
            "What does this project do and how is it organised?",
            "Where is the main entry point and what happens on startup?",
            "Which files handle configuration and environment variables?",
            "How is error handling done across the code?",
        ]
        c1, c2 = st.columns(2)
        for i, q in enumerate(starters):
            (c1 if i % 2 == 0 else c2).button(q, key=f"starter_{i}", on_click=ask, args=(q,))

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m["role"] == "assistant" and show_sources:
                render_sources(m.get("sources", []), m.get("queries", []))

    if prompt and client and not allow_request():
        prompt = None

    if prompt:
        if not client:
            st.warning("Add your Groq API key in the sidebar to ask questions.")
            st.stop()

        history = [{"role": m["role"], "content": m["content"]} for m in messages]
        messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("Searching the codebase…", expanded=False) as status:
                queries = decompose_query(client, prompt, history)
                snippets = rag.retrieve(repo_id, queries, sources, prompt, top_k=top_k, max_snippets=max_snippets)
                status.update(label=f"Read {len(snippets)} snippets from {len({s['metadata']['source'] for s in snippets})} files", state="complete")

            try:
                answer = st.write_stream(
                    rag.stream_chat(client, model, rag.build_messages(tree, snippets, prompt, history), temperature)
                )
            except Exception as e:
                answer = explain_error(e)
                st.error(answer)

            if show_sources:
                render_sources(snippets, queries)

        messages.append({"role": "assistant", "content": answer, "sources": snippets, "queries": queries})

    if messages:
        c1, c2, _ = st.columns([1, 1, 4])
        c1.download_button("Export chat", chat_markdown(messages), file_name=f"{repo_id.replace('/', '_')}_chat.md")
        if c2.button("Clear chat"):
            ss["chats"][repo_id] = []
            st.rerun()

# ---------------------------------------------------------------- file map
elif view == "🗂️ File map":
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**Structure**")
        st.code(tree, language="text", height=520)
    with right:
        st.markdown("**Languages**")
        lang_df = pd.DataFrame({"files": lang_counts}).sort_values("files", ascending=False)
        st.bar_chart(lang_df, horizontal=True, height=220)
        st.markdown("**Files**")
        q = st.text_input("Filter", placeholder="Filter by path…", label_visibility="collapsed")
        df = pd.DataFrame(
            [{"File": f["source"], "Language": language_label(f["source"]), "Lines": f["lines"], "Chunks": f["chunks"]} for f in files]
        )
        if q:
            df = df[df["File"].str.contains(q, case=False, regex=False)]
        st.dataframe(df, hide_index=True, height=260)

# ---------------------------------------------------------------- briefing
else:
    st.markdown("A one-page orientation for someone new to the project, written from the README, config files and entry points.")
    summary = ss["summaries"].get(repo_id)
    if st.button("Regenerate briefing" if summary else "Generate briefing", type="primary"):
        if not client:
            st.warning("Add your Groq API key in the sidebar first.")
        elif allow_request():
            try:
                with st.spinner("Reading key files…"):
                    msgs = rag.overview_messages(repo_id, tree, sources)
                summary = st.write_stream(rag.stream_chat(client, model, msgs, 0.2))
                ss["summaries"][repo_id] = summary
            except Exception as e:
                st.error(explain_error(e))
    elif summary:
        st.markdown(summary)
    if ss["summaries"].get(repo_id):
        st.download_button("Download briefing", ss["summaries"][repo_id], file_name=f"{repo_id.replace('/', '_')}_briefing.md")
