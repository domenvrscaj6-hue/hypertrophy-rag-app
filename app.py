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
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
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
    
    st.subheader("⚙️ System Status")
    st.write(f"**Engine:** FAISS")
    st.write(f"**Chunk Size:** 1200")
    st.write(f"**Overlap:** 200")
    
    if st.button("🔄 Clear Cache", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.subheader("🌍 Translation")
    target_lang = st.selectbox("Translate Results to:", ["Slovenian", "German", "Spanish", "French", "Italian"], index=0)
    lang_map = {"Slovenian": "sl", "German": "de", "Spanish": "es", "French": "fr", "Italian": "it"}

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
    
    # Using text_input for free search, and selectbox for suggestions separately
    st.subheader("🔍 Search Research")
    
    # 1. Suggestions List
    suggestions = [
        "Select a pre-defined question...",
        "Does sleep affect muscle growth?", 
        "What is the optimal protein intake?", 
        "Is creatine supplementation safe for kidneys?", 
        "How does Vitamin D affect muscle health?"
    ]
    
    selected_suggestion = st.selectbox("Suggestions:", options=suggestions)
    
    # 2. Main Search Input (Works with Enter)
    query = st.text_input(
        "Or type your own question and press Enter:",
        value="" if selected_suggestion == suggestions[0] else selected_suggestion,
        placeholder="Type your question here...",
        key="main_search"
    )

    raw_docs, chunks, vectorstore = get_vectorstore()

    if query:
        with st.spinner(f"Searching for: '{query}'..."):
            results = vectorstore.similarity_search_with_relevance_scores(query, k=4)
            
            st.subheader(f"Results")
            
            if not results:
                st.warning("No matches found. Try different keywords.")
            else:
                for i, (doc, score) in enumerate(results):
                    if score < 0.1: continue
                    
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown')).replace('.txt', '.pdf')
                    char_count = len(doc.page_content)
                    
                    with st.expander(f"Result {i+1} (Relevance: {score:.2f})", expanded=(i==0)):
                        st.markdown(f"**English Original:**")
                        st.markdown(f"*{doc.page_content}*")
                        
                        if st.button(f"Translate to {target_lang}", key=f"tr_btn_{i}"):
                            translated = GoogleTranslator(source='auto', target=lang_map[target_lang]).translate(doc.page_content)
                            st.info(translated)
                        
                        st.divider()
                        c1, c2 = st.columns(2)
                        c1.caption(f"📍 **Source:** {source_file}")
                        c2.caption(f"📏 **Length:** {char_count} characters")
    else:
        st.info("Enter a question above to start searching.")

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
    * **Chunking:** 1200/200
    """)

st.sidebar.divider()
st.sidebar.caption("Built for AI Course • 2026")
