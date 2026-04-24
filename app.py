import streamlit as st
import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from deep_translator import GoogleTranslator

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hypertrophy RAG", page_icon="💪")

# --- API TOKEN ---
# Read from environment variable (set this in Render Dashboard)
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# --- FIXED PARAMETERS ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    
    # Use Hugging Face Inference API (Zero RAM usage on Render!)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return raw_documents, chunks, vectorstore

# --- APP UI ---
st.title("🏋️‍♂️ Hypertrophy Research Explorer")

if not HF_TOKEN:
    st.warning("⚠️ Please add your `HUGGINGFACEHUB_API_TOKEN` to Render Environment Variables to start.")
    st.stop()

# Pre-load knowledge base
with st.spinner("Connecting to scientific knowledge base via API..."):
    raw_docs, chunks, vectorstore = get_vectorstore()

# Navigation
menu = st.sidebar.radio("Menu", ["Search", "Stats", "About"])

if menu == "Search":
    st.subheader("🔍 Ask a research question")
    query = st.text_input("e.g., Is creatine safe for kidneys?", placeholder="Type your question...")
    
    if query:
        # We use k=5 for better results now that memory isn't an issue
        results = vectorstore.similarity_search_with_relevance_scores(query, k=5)
        
        if not results:
            st.warning("No relevant research found for this query.")
        else:
            for i, (doc, score) in enumerate(results):
                if score < 0.1: continue
                with st.expander(f"Result {i+1} (Relevance Score: {score:.2f})"):
                    st.write(doc.page_content)
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    st.caption(f"Source: {source_file}")
                    
                    if st.button(f"Translate Result {i+1} to Slovenian", key=f"tr_{i}"):
                        st.info(GoogleTranslator(source='auto', target='sl').translate(doc.page_content))

elif menu == "Stats":
    st.metric("Scientific Papers", len(raw_docs))
    st.metric("Text Segments", len(chunks))
    st.write("Average segment length:", int(sum(len(c.page_content) for c in chunks)/len(chunks)), "chars")
    
    source_counts = pd.Series([os.path.basename(c.metadata['source']) for c in chunks]).value_counts()
    st.bar_chart(source_counts)

elif menu == "About":
    st.write("### Technical Details")
    st.write("This app uses a Cloud-based RAG architecture:")
    st.write("- **Embeddings:** Hugging Face Inference API (`all-MiniLM-L6-v2`)")
    st.write("- **Vector Store:** FAISS (Facebook AI Similarity Search)")
    st.write("- **Processing:** 800-char chunks with 100-char overlap")
    st.write("This setup ensures high performance even on resource-constrained hosting like Render Free Tier.")
