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
st.set_page_config(page_title="Hypertrophy RAG", page_icon="💪")

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
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    
    # Modern Endpoint Embeddings with Retry logic
    embeddings = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=HF_TOKEN,
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Retry logic if model is loading on HF side
    max_retries = 3
    for attempt in range(max_retries):
        try:
            vectorstore = FAISS.from_documents(chunks, embeddings)
            return raw_documents, chunks, vectorstore
        except Exception as e:
            if attempt < max_retries - 1:
                st.warning(f"Hugging Face model is warming up... waiting 10s (Attempt {attempt+1}/{max_retries})")
                time.sleep(10)
            else:
                st.error(f"Error connecting to Hugging Face: {str(e)}")
                st.stop()

# --- APP UI ---
st.title("🏋️‍♂️ Hypertrophy Research Explorer")

if not HF_TOKEN:
    st.warning("⚠️ Please add your `HUGGINGFACEHUB_API_TOKEN` to Render Environment Variables.")
    st.stop()

with st.spinner("Connecting to scientific knowledge base..."):
    raw_docs, chunks, vectorstore = get_vectorstore()

menu = st.sidebar.radio("Menu", ["Search", "Stats", "About"])

if menu == "Search":
    st.subheader("🔍 Ask a research question")
    query = st.text_input("e.g., Is caffeine good for performance?", placeholder="Type your question...")
    
    if query:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=5)
        
        if not results:
            st.warning("No relevant research found.")
        else:
            for i, (doc, score) in enumerate(results):
                if score < 0.1: continue
                with st.expander(f"Result {i+1} (Score: {score:.2f})"):
                    st.write(doc.page_content)
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown'))
                    st.caption(f"Source: {source_file}")
                    
                    if st.button(f"Translate Result {i+1}", key=f"tr_{i}"):
                        st.info(GoogleTranslator(source='auto', target='sl').translate(doc.page_content))

elif menu == "Stats":
    st.metric("Scientific Papers", len(raw_docs))
    st.metric("Text Segments", len(chunks))
    source_counts = pd.Series([os.path.basename(c.metadata['source']) for c in chunks]).value_counts()
    st.bar_chart(source_counts)

elif menu == "About":
    st.write("Optimized Cloud-based RAG using Hugging Face API and FAISS.")
