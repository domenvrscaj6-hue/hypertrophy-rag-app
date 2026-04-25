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

# --- CLEAN UI STYLING ---
st.markdown("""
<style>
    [data-testid="metric-container"] {
        background-color: #f0f7ff;
        border: 2px solid #1e3a8a;
        padding: 15px;
        border-radius: 10px;
    }
    [data-testid="stMetricValue"] {
        color: #1e3a8a !important;
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
    
    # FINAL PRODUCTION SETTINGS: 1000 / 200
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=[". ", "\n\n", "\n", " ", ""]
    )
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

# --- CALLBACK FOR AUTOFILL ---
def set_query():
    if st.session_state.suggestion_box != "Select a suggestion...":
        st.session_state.user_query = st.session_state.suggestion_box

# --- SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("💪 Hypertrophy Lab")
    page = st.radio("Navigation", ["Home", "Search Knowledge Base", "Statistics", "About"], label_visibility="collapsed")
    st.divider()
    
    st.subheader("⚙️ System Status")
    st.write(f"**Engine:** FAISS")
    st.write(f"**Chunk Size:** 1000")
    st.write(f"**Overlap:** 200")

    st.divider()
    st.subheader("🌍 Translation")
    target_lang = st.selectbox("Translate Results to:", ["Slovenian", "German", "Spanish", "French", "Italian"], index=0)
    lang_map = {"Slovenian": "sl", "German": "de", "Spanish": "es", "French": "fr", "Italian": "it"}

    st.divider()
    st.caption("AI Course • Project Prototype • 2026")

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
    
    suggestions = [
        "Select a suggestion...",
        "Does sleep affect muscle growth?", 
        "What is the optimal protein intake?", 
        "Is creatine supplementation safe for kidneys?", 
        "How does Vitamin D affect muscle health?",
        "Effects of caffeine on athletic performance"
    ]
    
    st.selectbox("Suggested questions:", options=suggestions, key="suggestion_box", on_change=set_query)
    query = st.text_input("Ask your own research question:", key="user_query", placeholder="Type here and press Enter...")

    raw_docs, chunks, vectorstore = get_vectorstore()

    if query and query != "":
        with st.spinner(f"Searching..."):
            results = vectorstore.similarity_search_with_relevance_scores(query, k=4)
            st.subheader(f"Results")
            
            for i, (doc, score) in enumerate(results):
                if score < 0.05: continue 
                with st.expander(f"Result {i+1} (Relevance: {score:.2f})", expanded=(i==0)):
                    clean_content = doc.page_content.strip()
                    if not clean_content[0].isupper():
                        clean_content = "..." + clean_content
                    st.markdown(f"**English Original:**")
                    st.markdown(f"*{clean_content}*")
                    if st.button(f"Translate to {target_lang}", key=f"tr_btn_{i}"):
                        st.info(GoogleTranslator(source='auto', target=lang_map[target_lang]).translate(doc.page_content))
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.caption(f"📍 **Source:** {source_file if 'source_file' in locals() else os.path.basename(doc.metadata.get('source', 'Unknown'))}")
                    c2.caption(f"📏 **Length:** {len(doc.page_content)} characters")
    else:
        st.info("Start typing a question above to explore the research.")

# --- PAGE: STATISTICS ---
elif page == "Statistics":
    st.title("📊 Knowledge Base Insights")
    raw_docs, chunks, _ = get_vectorstore()
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents", len(set([doc.metadata.get('source') for doc in raw_docs])))
    m2.metric("Total Segments", len(chunks))
    m3.metric("Avg Segment", f"{int(sum(len(c.page_content) for c in chunks)/len(chunks))} chars")
    st.subheader("Data Distribution")
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
    * **Chunking:** 1000/200 (Sentence-aware splitting)
    """)

st.sidebar.divider()
st.sidebar.caption("Built for AI Course • 2026")
