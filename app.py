import streamlit as st
import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from deep_translator import GoogleTranslator

# --- PAGE CONFIG ---
st.set_page_config(page_title="Hypertrophy RAG", page_icon="💪")

# --- FIXED PARAMETERS (Saves memory by avoiding re-indexing) ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- CACHED DATA LOADING ---
@st.cache_resource
def get_vectorstore():
    data_path = "processed_texts/"
    if not os.path.exists(data_path):
        data_path = "streamlit-app/processed_texts/"
    
    loader = DirectoryLoader(data_path, glob="./*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    raw_documents = loader.load()
    
    # Clean hyphens/newlines common in PDFs
    for doc in raw_documents:
        doc.page_content = doc.page_content.replace("- ", "").replace("-\n", "")
        doc.page_content = " ".join(doc.page_content.split())
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(raw_documents)
    
    # Lightweight embeddings on CPU
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # FAISS is much more memory-efficient than Chroma
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    return raw_documents, chunks, vectorstore

# --- APP UI ---
st.title("🏋️‍♂️ Hypertrophy Research Explorer")

# Pre-load to verify status
with st.spinner("Loading scientific knowledge base..."):
    raw_docs, chunks, vectorstore = get_vectorstore()

# Navigation
menu = st.sidebar.radio("Menu", ["Search", "Stats", "About"])

if menu == "Search":
    st.subheader("🔍 Ask a research question")
    query = st.text_input("e.g., Is creatine safe for kidneys?", placeholder="Type your question...")
    
    if query:
        results = vectorstore.similarity_search_with_relevance_scores(query, k=3)
        for i, (doc, score) in enumerate(results):
            if score < 0.2: continue
            with st.expander(f"Result {i+1} (Score: {score:.2f})"):
                st.write(doc.page_content)
                if st.button(f"Translate to Slovenian", key=f"tr_{i}"):
                    st.info(GoogleTranslator(source='auto', target='sl').translate(doc.page_content))

elif menu == "Stats":
    st.metric("Scientific Papers", len(raw_docs))
    st.metric("Text Segments", len(chunks))
    st.write("Average segment length:", int(sum(len(c.page_content) for c in chunks)/len(chunks)), "chars")

elif menu == "About":
    st.write("This app uses RAG (Retrieval-Augmented Generation) to search through hypertrophy research.")
    st.write(f"Settings: FAISS, Chunk Size {CHUNK_SIZE}, Overlap {CHUNK_OVERLAP}.")
