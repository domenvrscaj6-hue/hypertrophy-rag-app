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
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .stAlert { border-radius: 10px; }
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
    
    # We keep your chosen optimized parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Retry logic for API warm-up
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
    st.subheader("⚙️ Search Strategy")
    st.info("Strategy: **Semantic RAG**\n\nChunk Size: **800**\n\nOverlap: **100**")
    
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

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
        st.markdown(f"""
        ### Turn Research into Results.
        This AI-powered knowledge base uses **Semantic Search** to scan peer-reviewed literature 
        on muscle growth, nutrition, and exercise physiology.
        """)
        st.success("✅ Application is LIVE and connected to Cloud Research Database.")
    with col2:
        st.image("https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=1470&auto=format&fit=crop", use_container_width=True)

# --- PAGE: SEARCH ---
elif page == "Search Knowledge Base":
    st.title("🔍 Research Retrieval")
    
    suggestions = [
        "Does sleep deprivation affect muscle growth?",
        "What is the optimal protein intake for hypertrophy?",
        "Is creatine supplementation safe for kidneys?",
        "How does Vitamin D affect muscle health?",
        "Effects of caffeine on athletic performance",
        "Role of rehydration in endurance exercise"
    ]
    
    query = st.selectbox(
        "Search or select a suggested question:",
        options=suggestions,
        index=None,
        placeholder="Type your question here...",
        help="Start typing to see suggestions!"
    )

    with st.spinner("Accessing knowledge base..."):
        raw_docs, chunks, vectorstore = get_vectorstore()

    if query:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=4)
        
        st.subheader(f"Results for: '{query}'")
        
        if not results:
            st.warning("No matches found in the current research database.")
        else:
            for i, (doc, score) in enumerate(results):
                if score < 0.1: continue
                
                source_file = os.path.basename(doc.metadata.get('source', 'Unknown')).replace('.txt', '.pdf')
                
                with st.expander(f"Result {i+1} (Relevance: {score:.2f})", expanded=(i==0)):
                    st.markdown(f"**English Original:**")
                    st.markdown(f"*{doc.page_content}*")
                    
                    if st.checkbox(f"Translate to {target_lang}", key=f"trans_{i}"):
                        translated = GoogleTranslator(source='auto', target=lang_map[target_lang]).translate(doc.page_content)
                        st.success(translated)
                    
                    st.divider()
                    c1, c2 = st.columns(2)
                    c1.caption(f"📍 **Source:** {source_file}")
                    c2.caption(f"📏 **Length:** {len(doc.page_content)} chars")
    else:
        st.info("Start typing a question above to explore the research.")

# --- PAGE: STATISTICS ---
elif page == "Statistics":
    st.title("📊 Knowledge Base Insights")
    raw_docs, chunks, _ = get_vectorstore()
    
    m1, m2, m3 = st.columns(3)
    unique_sources = len(set([doc.metadata.get('source') for doc in raw_docs]))
    m1.metric("Documents", unique_sources)
    m2.metric("Total Segments", len(chunks))
    m3.metric("Avg Segment", f"{int(sum(len(c.page_content) for c in chunks)/len(chunks))} chars")

    st.subheader("Data Distribution")
    source_counts = pd.Series([os.path.basename(c.metadata.get('source')) for c in chunks]).value_counts()
    st.bar_chart(source_counts)

# --- PAGE: ABOUT ---
elif page == "About":
    st.title("ℹ️ Technical Overview")
    st.markdown(f"""
    This app uses **RAG** (Retrieval-Augmented Generation) with surgically clean scientific data.
    
    **Cloud-Optimized Architecture:**
    * **Embeddings:** Hugging Face Inference API (`all-MiniLM-L6-v2`)
    * **Vector Store:** FAISS (Facebook AI Similarity Search)
    * **Preprocessing:** Recursive character splitting (800/100)
    * **UI:** Streamlit 1.43
    """)

st.sidebar.divider()
st.sidebar.caption("Built for AI Course • April 2026")
