# app.py
import os
from pathlib import Path
from datetime import datetime, timedelta
import re
from dateutil import parser

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------
# ENV
# --------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY missing in environment")
    st.stop()

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
LOG_FOLDER = "./logs"
INDEX_PATH = "faiss_index"
LAST_N_DAYS = 10

# Patterns to detect timestamps in logs
DATE_PATTERNS = [
    r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)-\d{4}\s?\d{2}:?\d{2}?",   # Nov-2020 1425
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}",                                         # 2026-02-20 14:25:30
    r"\d{2}/\d{2}/\d{4}\s\d{2}:\d{2}",                                                 # 20/02/2026 14:25
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{1,2},\s\d{4}\s\d{2}:\d{2}" # Feb 20, 2026 14:25
]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="AI Log Analyzer", layout="wide")
st.title("🧠 AI Log Root Cause Analyzer (Last 15 Days Logs)")

# --------------------------------------------------
# Load Logs with Date Filtering
# --------------------------------------------------
def load_logs():
    """
    Load log files and filter only lines from last N days
    """

    if not Path(LOG_FOLDER).exists():
        return []

    loader = DirectoryLoader(
        path=LOG_FOLDER,
        glob="**/*.[tl][xo][gt]",     # .txt and .log
        loader_cls=TextLoader,
        loader_kwargs={"autodetect_encoding": True},
        show_progress=True
    )

    docs = loader.load()

    cutoff_date = datetime.now() - timedelta(days=LAST_N_DAYS)
    filtered_docs = []

    for doc in docs:
        lines = doc.page_content.splitlines()
        filtered_lines = []

        for line in lines:
            line_date = None
            for pattern in DATE_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    try:
                        line_date = parser.parse(match.group(), fuzzy=True)
                        break
                    except Exception:
                        continue
            if line_date and line_date >= cutoff_date:
                filtered_lines.append(line)

        if filtered_lines:
            doc.page_content = "\n".join(filtered_lines)
            filtered_docs.append(doc)

    return filtered_docs

# --------------------------------------------------
# Build / Load Vector DB
# --------------------------------------------------
@st.cache_resource(show_spinner="Building vector index...")
def build_vector_db(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )

    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(INDEX_PATH)
    return db

def load_vector_db():
    if not os.path.exists(INDEX_PATH):
        return None
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=OPENAI_API_KEY
    )
    return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    rebuild = st.button("🔄 Rebuild Index")
    st.markdown("---")
    st.write("📁 Log Folder")
    st.code(LOG_FOLDER)

# --------------------------------------------------
# Load Logs
# --------------------------------------------------
docs = load_logs()
if not docs:
    st.error(f"❌ No logs found for last {LAST_N_DAYS} days")
    st.stop()

# --------------------------------------------------
# Vector DB Handling
# --------------------------------------------------
vector_db = None
if rebuild:
    st.cache_resource.clear()
    vector_db = build_vector_db(docs)
    st.success("✅ Index rebuilt")
else:
    vector_db = load_vector_db()
    if not vector_db:
        vector_db = build_vector_db(docs)

retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=1200,
    api_key=OPENAI_API_KEY
)

# --------------------------------------------------
# RCA Prompt
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system",
     """
You are a senior software reliability engineer.
Analyze application logs and do the following:
1. Identify main error(s)
2. Find root cause
3. Correlate timestamps if needed
4. Explain failure chain
5. Recommend technical solution
6. Suggest prevention steps
Use only provided logs. Be precise and technical.
"""),
    ("human",
     """
Logs:
{context}

Question:
{question}
""")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --------------------------------------------------
# Chain
# --------------------------------------------------
chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------------------------------
# Chat Interface
# --------------------------------------------------
st.subheader("💬 Ask About System Issues")
question = st.text_input("Example: Why did service crash on Feb 15?")

if question:
    with st.spinner("Analyzing logs..."):
        docs_found = retriever.invoke(question)
        st.caption(f"🔎 Related log sections: {len(docs_found)}")
        answer = chain.invoke(question)
    st.markdown("## 📊 Root Cause Analysis")
    st.write(answer)