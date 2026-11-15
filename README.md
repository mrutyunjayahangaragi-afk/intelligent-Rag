# 🏢 Zyro Dynamics · HR Intelligence Assistant (Intelligent RAG)

An enterprise-grade, high-precision Retrieval-Augmented Generation (RAG) assistant designed to instantly and accurately answer company policy, benefits, and compliance questions grounded directly in official internal HR documentation.

Built with **Streamlit**, **LangChain**, **Groq LPU (GPT-OSS / LLaMA / Qwen)**, **FAISS Vector Database**, and **HuggingFace Embeddings**.

---

## 🌟 Key Features

- **Grounded & Zero-Hallucination Answers**: System rules ensure the assistant only answers when verifiable evidence exists in the indexed policy documents, returning exact numbers, eligibility tiers, caps, and dates.
- **Auditable Source Attribution**: Every statement is cited with clickable policy references displaying the source document name, exact page number, and relevant excerpt.
- **Enterprise-Grade Modern UI**:
  - Clean corporate emerald design system with real-time knowledge base statistics.
  - Interactive starter prompt cards for common topics (Annual Leave, WFH, Compensation, Performance Review).
  - Responsive chat interface with Markdown tables, bold bullet points, and code callouts.
- **Dynamic Policy Ingestion**:
  - In-browser PDF uploader: Drag-and-drop or upload new policy documents directly from the UI.
  - Live document catalog viewer displaying page counts and file sizes.
- **Persistent Local Caching**:
  - Automatically caches the FAISS vector index to disk (`.faiss_cache`) using MD5 fingerprinting, enabling instant restarts without re-embedding.
- **Inference & Retrieval Controls**:
  - Configurable Groq LLM engine (`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, etc.).
  - Adjustable MMR retrieval depth ($k = 4$ to $14$ chunks) and response temperature.
- **Export & Management**:
  - Export full chat transcripts directly to Markdown.
  - One-click cache refresh and chat reset.

---

## 📁 Indexed HR Policy Coverage

The knowledge base covers 11 core organizational policy domains:
1. `00_Company_Profile.pdf` – Company background, culture, and organizational hierarchy.
2. `01_Employee_Handbook.pdf` – General workplace policies and employee expectations.
3. `02_Leave_Policy.pdf` – Earned, casual, sick, maternity, paternity, and bereavement leave entitlements.
4. `03_Work_From_Home_Policy.pdf` – Remote work eligibility, core hours, and approval workflows.
5. `04_Code_of_Conduct.pdf` – Business ethics, gift policies, and conflict of interest guidelines.
6. `05_Performance_Review_Policy.pdf` – APR timelines, rating scales, and PIP criteria.
7. `06_Compensation_and_Benefits_Policy.pdf` – Grade CTC ranges, medical insurance, and ESOP vesting schedules.
8. `07_IT_and_Data_Security_Policy.pdf` – Password standards, MFA requirements, and device compliance.
9. `08_Prevention_of_Sexual_Harassment_Policy.pdf` – POSH procedures, ICC timelines, and safeguards.
10. `09_Onboarding_and_Separation_Policy.pdf` – Probation periods, resignation notice requirements, and clearance procedures.
11. `10_Travel_and_Expense_Policy.pdf` – Grade-based domestic and international travel entitlements.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- [Groq API Key](https://console.groq.com)

### 2. Installation
Clone the repository and set up a virtual environment:
```bash
git clone https://github.com/mrutyunjayahangaragi-afk/intelligent-Rag.git
cd intelligent-Rag

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Secrets
Create a `.streamlit/secrets.toml` file (or copy from `.streamlit/secrets.toml.example`):
```toml
GROQ_API_KEY = "your_groq_api_key_here"
GROQ_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
```

### 4. Run the Application
Launch the Streamlit app:
```bash
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
Streamlit UI  ──►  MMR Retriever (k=8, λ=0.80)
                         │
                         ▼
             FAISS Vector Store (.faiss_cache)
             (HuggingFace all-MiniLM-L6-v2)
                         │
                         ▼
             Extracted Context Passages + [Doc, Page]
                         │
                         ▼
             Groq API (openai/gpt-oss-120b LPU)
                         │
                         ▼
             Grounded Markdown Response + Verified Citations
```

---

## 🔒 Security & Privacy
- API keys are handled strictly via Streamlit secrets or environment variables.
- Sensitive credentials and local cache directories (`.faiss_cache/`, `.venv/`, `.streamlit/secrets.toml`) are excluded in `.gitignore`.
