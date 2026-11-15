"""Streamlit Enterprise UI for the Zyro Dynamics HR Intelligent RAG Assistant."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import streamlit as st

from rag_engine import (
    AVAILABLE_GROQ_MODELS,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GROQ_MODEL,
    RAGConfigurationError,
    build_rag_engine,
    get_document_catalog,
)

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "hr_policies"
PDF_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Zyro Dynamics · HR Intelligence",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Custom Modern Corporate CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

      html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      }
      
      .stApp {
        background: #f8faf9;
        color: #1a2e2b;
      }

      /* Header Bar */
      [data-testid="stHeader"] {
        background: rgba(248, 250, 249, 0.85);
        backdrop-filter: blur(8px);
      }

      /* Main container padding */
      .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
      }

      /* Hero Banner */
      .zyro-hero {
        background: linear-gradient(135deg, #0e3b32 0%, #175d4f 55%, #1d7262 100%);
        border-radius: 20px;
        padding: 1.8rem 2.2rem;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 10px 25px -5px rgba(18, 64, 54, 0.22);
        position: relative;
        overflow: hidden;
      }
      .zyro-hero::after {
        content: "";
        position: absolute;
        top: -40px;
        right: -40px;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(255,255,255,0.12) 0%, rgba(255,255,255,0) 70%);
        border-radius: 50%;
      }
      .zyro-hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.25);
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        color: #e2f4ee;
      }
      .zyro-hero h1 {
        color: #ffffff !important;
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 0.4rem 0;
      }
      .zyro-hero p {
        color: #d1e7e0;
        font-size: 1rem;
        margin: 0;
        max-width: 720px;
        line-height: 1.5;
      }

      /* Metrics Bar */
      .stats-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.8rem;
        margin-bottom: 1.5rem;
      }
      .stat-chip {
        flex: 1 1 180px;
        background: #ffffff;
        border: 1px solid #e3eae7;
        border-radius: 14px;
        padding: 0.85rem 1.1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        transition: transform 0.15s ease;
      }
      .stat-chip:hover {
        transform: translateY(-2px);
      }
      .stat-chip-label {
        font-size: 0.76rem;
        color: #647d76;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 600;
        margin-bottom: 0.2rem;
      }
      .stat-chip-val {
        font-size: 1.35rem;
        font-weight: 700;
        color: #113830;
      }

      /* Starter Prompt Cards */
      .starter-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
        gap: 0.9rem;
        margin-bottom: 1.8rem;
      }
      .starter-card {
        background: #ffffff;
        border: 1px solid #e1e9e5;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        cursor: pointer;
        transition: all 0.2s ease;
        box-shadow: 0 2px 6px rgba(0,0,0,0.02);
      }
      .starter-card:hover {
        border-color: #176b57;
        box-shadow: 0 6px 16px rgba(23, 107, 87, 0.08);
        transform: translateY(-2px);
      }
      .starter-icon {
        font-size: 1.3rem;
        margin-bottom: 0.5rem;
      }
      .starter-title {
        font-size: 0.92rem;
        font-weight: 600;
        color: #153e35;
        margin-bottom: 0.25rem;
      }
      .starter-desc {
        font-size: 0.8rem;
        color: #687e77;
        line-height: 1.35;
      }

      /* Chat Messages */
      div[data-testid="stChatMessage"] {
        background: #ffffff;
        border: 1px solid #e6ece9;
        border-radius: 16px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
      }
      
      /* Source citation box */
      .source-box {
        background: #f2f7f5;
        border: 1px solid #dbe6e2;
        border-radius: 10px;
        padding: 0.65rem 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.5rem;
      }
      .source-tag {
        display: inline-block;
        background: #176b57;
        color: white;
        font-size: 0.72rem;
        font-weight: 600;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 6px;
      }

      /* Tables in responses */
      table {
        border-collapse: collapse;
        width: 100%;
        margin: 0.8rem 0;
        font-size: 0.9rem;
      }
      th {
        background-color: #e9f2ef !important;
        color: #103830 !important;
        font-weight: 600;
        text-align: left;
        padding: 8px 12px;
        border: 1px solid #d4e0dc;
      }
      td {
        padding: 8px 12px;
        border: 1px solid #e3ece8;
      }
      tr:nth-child(even) {
        background-color: #fbfdfc;
      }

      /* Sidebar polish */
      [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e7ecea;
      }
      .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
      }
      .sidebar-brand h2 {
        font-size: 1.25rem;
        font-weight: 800;
        color: #123b34;
        margin: 0;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


# ---------------------------------------------------------
# Sidebar UI & Knowledge Base Explorer
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <span style="font-size: 1.8rem;">🏢</span>
          <div>
            <h2 style="margin:0; font-size:1.2rem; color:#123b34;">Zyro Dynamics</h2>
            <span style="font-size:0.75rem; color:#688078; font-weight:600; text-transform:uppercase;">HR Intelligence Desk</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="background:#e8f4f0; border-left:4px solid #176b57; padding:8px 12px; border-radius:6px; font-size:0.8rem; color:#14453b; margin-bottom:1rem;">
          🟢 <b>System Active</b> · Zero-Hallucination Grounded Policy Q&A
        </div>
        """,
        unsafe_allow_html=True,
    )

    tabs = st.tabs(["📚 Knowledge Base", "⚙️ Tuning", "💬 Options"])

    with tabs[0]:
        st.markdown("#### Policy Documents")
        catalog = get_document_catalog(PDF_DIR)
        
        if catalog:
            st.caption(f"Currently indexing **{len(catalog)} official policy files**:")
            with st.container(height=260):
                for item in catalog:
                    with st.expander(f"📄 {item['name']}", expanded=False):
                        st.write(f"- **Pages:** {item['pages']}")
                        st.write(f"- **File size:** {item['size_kb']} KB")
        else:
            st.warning("No PDF policy files found in knowledge base.")

        st.divider()
        st.markdown("#### Upload Policy Documents")
        uploaded_files = st.file_uploader(
            "Add or update HR policies (PDF)",
            type=["pdf"],
            accept_multiple_files=True,
            help="Upload new policy documents here to instantly integrate them into the RAG vector store.",
        )
        if uploaded_files:
            new_added = 0
            for ufile in uploaded_files:
                save_dest = PDF_DIR / ufile.name
                with open(save_dest, "wb") as f:
                    f.write(ufile.getbuffer())
                new_added += 1
            if new_added > 0:
                st.success(f"Uploaded {new_added} document(s). Rebuilding index...")
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()

    with tabs[1]:
        st.markdown("#### Model & Retrieval Parameters")
        
        configured_model = get_secret("GROQ_MODEL", DEFAULT_GROQ_MODEL) or DEFAULT_GROQ_MODEL
        model_options = list(dict.fromkeys([configured_model] + AVAILABLE_GROQ_MODELS))
        
        selected_model = st.selectbox(
            "LLM Engine (Groq)",
            options=model_options,
            index=0,
            help="Select the language model running on Groq LPU inference.",
        )

        selected_top_k = st.slider(
            "Retrieval Depth (Chunks)",
            min_value=4,
            max_value=14,
            value=8,
            step=1,
            help="Number of most relevant policy passages retrieved per question using MMR search.",
        )

        selected_temp = st.slider(
            "Temperature (Creativity vs Factuality)",
            min_value=0.0,
            max_value=0.4,
            value=0.0,
            step=0.05,
            help="Keep at 0.0 for strict factual policy compliance.",
        )

    with tabs[2]:
        st.markdown("#### Conversation")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        if st.button("🔄 Rebuild Knowledge Base", use_container_width=True):
            st.cache_resource.clear()
            st.success("Vector cache cleared. Refreshing...")
            time.sleep(0.8)
            st.rerun()

        if "messages" in st.session_state and st.session_state.messages:
            transcript = "\n\n".join(
                f"**{m['role'].title()}**: {m['content']}" for m in st.session_state.messages
            )
            st.download_button(
                "📥 Export Chat (Markdown)",
                data=transcript,
                file_name="zyro_hr_conversation.md",
                mime="text/markdown",
                use_container_width=True,
            )


# ---------------------------------------------------------
# Load Knowledge Base & Engine
# ---------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_cached_engine(
    pdf_dir: str,
    groq_api_key: str,
    groq_model: str,
    embedding_model: str,
):
    return build_rag_engine(
        pdf_dir=pdf_dir,
        groq_api_key=groq_api_key,
        groq_model=groq_model,
        embedding_model=embedding_model,
    )


api_key = get_secret("GROQ_API_KEY")
groq_model = selected_model
embedding_model = get_secret("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL

# ---------------------------------------------------------
# Main Page Header & Hero
# ---------------------------------------------------------
st.markdown(
    """
    <div class="zyro-hero">
      <div class="zyro-hero-badge">🏢 Official HR Assistant · Zyro Dynamics</div>
      <h1>HR Policy Intelligence Desk</h1>
      <p>Instant, verifiable answers grounded directly in official Zyro Dynamics employment handbooks, compensation guidelines, and internal policies.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Check API Key
if not api_key:
    st.error(
        "⚠️ **Groq API key not found.** Please configure `GROQ_API_KEY` in `.streamlit/secrets.toml` "
        "or in the app environment."
    )
    st.stop()

# Initialize Engine
engine = None
pdf_files = list(PDF_DIR.rglob("*.pdf")) if PDF_DIR.exists() else []

if not pdf_files:
    st.warning("⚠️ **No HR policy PDFs detected in `data/hr_policies/`.**")
    st.info("Use the file uploader in the left sidebar to add policy PDFs, or place them in `data/hr_policies/`.")
    st.stop()

try:
    with st.spinner("Connecting to HR Knowledge Base & Vector Index..."):
        engine = load_cached_engine(
            str(PDF_DIR),
            api_key,
            groq_model,
            embedding_model,
        )
except RAGConfigurationError as exc:
    st.error(f"Configuration Error: {exc}")
    st.stop()
except Exception as exc:
    st.error(f"Knowledge Base initialization error: {exc}")
    st.stop()

# Display Live Knowledge Base Stats Strip
st.markdown(
    f"""
    <div class="stats-strip">
      <div class="stat-chip">
        <div class="stat-chip-label">Indexed Policies</div>
        <div class="stat-chip-val">📚 {engine.pdf_count}</div>
      </div>
      <div class="stat-chip">
        <div class="stat-chip-label">Extracted Pages</div>
        <div class="stat-chip-val">📄 {engine.page_count}</div>
      </div>
      <div class="stat-chip">
        <div class="stat-chip-label">Semantic Chunks</div>
        <div class="stat-chip-val">🧩 {engine.chunk_count}</div>
      </div>
      <div class="stat-chip">
        <div class="stat-chip-label">Inference Model</div>
        <div class="stat-chip-val" style="font-size:1.05rem; padding-top:4px;">⚡ {groq_model.split('/')[-1]}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Chat Session State & Starter Suggestions
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# If empty chat, show helpful starter prompt cards
if not st.session_state.messages:
    st.markdown("### 💡 Frequently Asked Policy Topics")
    st.caption("Click any topic below to run the inquiry instantly, or type your own question in the chat bar.")

    col1, col2, col3, col4 = st.columns(4)

    prompt_to_send = None

    with col1:
        st.markdown(
            """
            <div class="starter-card">
              <div class="starter-icon">🏖️</div>
              <div class="starter-title">Annual Leave & Carry Forward</div>
              <div class="starter-desc">Check Earned Leave accrual rate, probation rules, and 45-day caps.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ask Leave Entitlement", key="btn_q1", use_container_width=True):
            prompt_to_send = "What is the annual leave entitlement and carry forward policy?"

    with col2:
        st.markdown(
            """
            <div class="starter-card">
              <div class="starter-icon">💻</div>
              <div class="starter-title">Work-From-Home (WFH)</div>
              <div class="starter-desc">Who is eligible for remote work, grade rules, and approval requirements.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ask WFH Eligibility", key="btn_q2", use_container_width=True):
            prompt_to_send = "What are the eligibility criteria and approval process for working from home?"

    with col3:
        st.markdown(
            """
            <div class="starter-card">
              <div class="starter-icon">💰</div>
              <div class="starter-title">Compensation & Benefits</div>
              <div class="starter-desc">Salary credit dates, Grade L4 CTC brackets, and group insurance limits.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ask Salary & Insurance", key="btn_q3", use_container_width=True):
            prompt_to_send = "When is salary credited, what is the L4 CTC range, and what medical insurance is provided?"

    with col4:
        st.markdown(
            """
            <div class="starter-card">
              <div class="starter-icon">📈</div>
              <div class="starter-title">Performance & PIP</div>
              <div class="starter-desc">Annual performance review schedule (APR) and PIP trigger thresholds.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Ask Performance & PIP", key="btn_q4", use_container_width=True):
            prompt_to_send = "What triggers a Performance Improvement Plan (PIP) and what is the Annual Performance Review timeline?"

    if prompt_to_send:
        st.session_state.pending_query = prompt_to_send
        st.rerun()

# ---------------------------------------------------------
# Render Chat History
# ---------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🏢"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📑 Verified Citations ({len(msg['sources'])} policy references)", expanded=False):
                for idx, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"""
                        <div class="source-box">
                          <span class="source-tag">Reference {idx}</span>
                          <b>{src['source']}</b> · <span style="color:#57736b;">Page {src['page']}</span>
                          <div style="font-size:0.83rem; color:#495f59; margin-top:5px; line-height:1.4;">
                            “{src['excerpt']}...”
                          </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------
# User Query Input Handling
# ---------------------------------------------------------
active_query = None

# Check if triggered by starter card
if "pending_query" in st.session_state and st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None
else:
    active_query = st.chat_input("Ask any question regarding Zyro Dynamics HR policies...")

if active_query:
    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user", avatar="👤"):
        st.markdown(active_query)

    # Process Assistant Response
    with st.chat_message("assistant", avatar="🏢"):
        with st.spinner("Searching HR policies and generating grounded answer..."):
            try:
                t0 = time.time()
                result = engine.ask(
                    active_query,
                    top_k=selected_top_k,
                    temperature=selected_temp,
                    override_model=selected_model,
                )
                duration = round(time.time() - t0, 2)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    st.caption(f"⚡ Grounded from {len(sources)} policy passage(s) in {duration}s using {selected_model.split('/')[-1]}")
                    with st.expander(f"📑 Verified Citations ({len(sources)} policy references)", expanded=False):
                        for idx, src in enumerate(sources, 1):
                            st.markdown(
                                f"""
                                <div class="source-box">
                                  <span class="source-tag">Reference {idx}</span>
                                  <b>{src['source']}</b> · <span style="color:#57736b;">Page {src['page']}</span>
                                  <div style="font-size:0.83rem; color:#495f59; margin-top:5px; line-height:1.4;">
                                    “{src['excerpt']}...”
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                else:
                    st.caption(f"⚡ Completed in {duration}s · Zero document match / Out of policy scope")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                err_msg = (
                    "⚠️ **Unable to process question.** Please verify the Groq API key "
                    f"and network connectivity. Details: `{exc}`"
                )
                st.error(err_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": err_msg, "sources": []}
                )
