import streamlit as st
import pandas as pd
import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from deep_translator import GoogleTranslator

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Hypertrophy Science RAG",
    page_icon="💪",
    layout="wide"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8f9fa; color: #1e1e1e; }
    
    /* Metrics styling - forcing dark text and blue accents */
    [data-testid="stMetricValue"] {
        color: #1e3a8a !important; /* Dark Blue */
        font-weight: bold;
    }
    [data-testid="stMetricLabel"] {
        color: #333333 !important; /* Dark Grey/Black */
    }
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Typography */
    h1, h2, h3 { color: #1e3a8a !important; }
    p, li { color: #333333 !important; }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #ffffff !important;
        color: #1e3a8a !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- API TOKEN ---
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- CACHED DATA LOADING ---
@st.cache_resource
def get_vectorstore():
    if not HF_TOKEN:
        st.error("Missing Hugging Face API Token! Please set HUGGINGFACEHUB_API_TOKEN in environment variables.")
        st.stop()

    data_path = "processed_texts/"
    if not os.path.exists(data_path):
        data_path = "streamlit-app/processed_texts/"
    
    loader = DirectoryLoader(data_path, glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_documents = loader.load()
    
    for doc in raw_documents:
        doc.page_content = doc.page_content.replace("- ", "").replace("-\n", "")
        doc.page_content = " ".join(doc.page_content.split())
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = text_splitter.split_documents(raw_documents)
    
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            return raw_documents, chunks, vectorstore
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Connecting to Cloud Model... (Attempt {attempt+1}/{max_retries})")
                time.sleep(12)
            else:
                st.error(f"Cloud Connection Error: {str(e)}")
                st.stop()

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("💪 Hypertrophy Lab")
    page = st.radio("Navigation", ["Home", "Search Knowledge Base", "Statistics", "About"], label_visibility="collapsed")
    st.divider()
    st.info("**RAG Engine:** FAISS\n**Embeddings:** HF Cloud\n**Status:** Optimized")
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

# --- PAGE: HOME ---
if page == "Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.title("🏋️‍♂️ Evidence-Based Hypertrophy Explorer")
        st.markdown("""
        ### Turn Research into Results.
        This AI-powered knowledge base scans peer-reviewed literature 
        on muscle growth, nutrition, and exercise physiology.
        """)
        st.success("✅ Application is connected to Cloud Research Database.")
    with col2:
        st.image("https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=1470&auto=format&fit=crop", use_container_width=True)

# --- PAGE: SEARCH ---
elif page == "Search Knowledge Base":
    st.title("🔍 Research Retrieval")
    suggestions = ["Does sleep affect muscle growth?", "Optimal protein intake?", "Is creatine safe?", "Vitamin D and muscle?"]
    query = st.selectbox("Search or select a question:", options=suggestions, index=None, placeholder="Type here...")

    raw_docs, chunks, vectorstore = get_vectorstore()

    if query:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=4)
        st.subheader(f"Results for: '{query}'")
        for i, (doc, score) in enumerate(results):
            if score < 0.1: continue
            with st.expander(f"Result {i+1} (Score: {score:.2f})", expanded=(i==0)):
                st.markdown(f"*{doc.page_content}*")
                if st.button(f"Translate to Slovenian", key=f"tr_{i}"):
                    st.info(GoogleTranslator(source='auto', target='sl').translate(doc.page_content))
                st.caption(f"Source: {os.path.basename(doc.metadata.get('source', 'Unknown'))}")

# --- PAGE: STATISTICS ---
elif page == "Statistics":
    st.title("📊 Knowledge Base Insights")
    raw_docs, chunks, _ = get_vectorstore()
    
    m1, m2, m3 = st.columns(3)
    unique_sources = len(set([doc.metadata.get('source') for doc in raw_docs]))
    m1.metric("Documents", unique_sources)
    m2.metric("Total Segments", len(chunks))
    m3.metric("Avg Segment", f"{int(sum(len(c.page_content) for c in chunks)/len(chunks))} chars")

    st.subheader("Data Distribution (Segments per File)")
    source_counts = pd.DataFrame([os.path.basename(c.metadata.get('source')) for c in chunks], columns=["Source"]).value_counts().reset_index()
    source_counts.columns = ["Source", "Count"]
    st.bar_chart(source_counts.set_index("Source"))

# --- PAGE: ABOUT ---
elif page == "About":
    st.title("ℹ️ Technical Overview")
    st.markdown("""
    **Cloud-Optimized Architecture:**
    * **Embeddings:** Hugging Face API
    * **Vector Store:** FAISS
    * **Chunking:** 800/100
    """)

st.sidebar.divider()
st.sidebar.caption("AI Course • 2026")
