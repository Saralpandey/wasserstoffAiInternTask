# streamlit_app.py (FINAL with all 4 features)
import streamlit as st
import os
import chromadb
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import tempfile
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# SETUP
st.set_page_config(page_title="DocBot 🤖", layout="wide")
st.title("📄 Gen-AI Chatbot for Document Q&A")

# Groq API Key
GROQ_API_KEY = "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"

# Init Chroma client
persist_dir = "./vector_store"
chroma_client = chromadb.PersistentClient(path=persist_dir)
collection = chroma_client.get_or_create_collection(name="documents")

# OCR extractors
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(file):
    pages = convert_from_path(file, 300)
    return "\n".join([pytesseract.image_to_string(p) for p in pages])

@st.cache_data(show_spinner=False)
def extract_text_from_image(file):
    image = Image.open(file)
    return pytesseract.image_to_string(image)

# Upload UI
st.sidebar.header("📄 Upload Files")
uploaded_files = st.sidebar.file_uploader("Choose files (PDF or Images)", accept_multiple_files=True, type=['pdf', 'png', 'jpg', 'jpeg'])

# Extract and store data
docs = []
file_texts = []
doc_map = {}  # filename -> full extracted text
if uploaded_files:
    for file in uploaded_files:
        ext = file.name.split('.')[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext) as tmp:
            tmp.write(file.read())
            path = tmp.name

        text = extract_text_from_pdf(path) if ext == 'pdf' else extract_text_from_image(path)

        doc_id = file.name.replace(" ", "_")
        collection.add(documents=[text], ids=[doc_id], metadatas=[{"source": doc_id}])
        st.sidebar.success(f"✅ {file.name} processed")

        doc_map[file.name] = text
        file_texts.append((file.name, text))
        docs.append(Document(page_content=text, metadata={"source": doc_id}))

# Feature 1: OCR Text Preview
if file_texts:
    st.subheader("📝 Extracted Text from Files")
    for fname, txt in file_texts:
        with st.expander(f"📄 {fname}"):
            st.markdown(f"```\n{txt.strip()[:1500]}\n```)  # first 1500 chars")

# Feature 4: Auto Theme Tags (via Clustering)
def show_theme_tags(docs):
    st.subheader("🎨 Theme Tags from Documents")
    texts = [doc.page_content for doc in docs]
    if not texts: return

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)
    k = min(4, len(texts))
    model = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(X)
    themes = defaultdict(list)
    for i, label in enumerate(model.labels_):
        themes[f"Theme {label+1}"].append(docs[i].metadata['source'])

    for theme, files in themes.items():
        with st.expander(f"🪄 {theme}"):
            st.write("Related Files:")
            for f in files:
                st.markdown(f"- {f}")

show_theme_tags(docs)

# Feature 2 + 3: QA Chat + source highlighting
if docs:
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embed, persist_directory=persist_dir)

    llm = ChatGroq(api_key=GROQ_API_KEY, model_name="llama3-8b-8192")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), return_source_documents=True)

    st.subheader("💬 Ask a Question from Your Files")
    query = st.text_input("Enter your question:", placeholder="e.g. kis shayari me zakhm ka zikr hai?")

    if query:
        with st.spinner("🤖 Thinking..."):
            result = qa(query)
            st.markdown(f"### 🤖 Answer:\n{result['result']}")
            st.markdown("#### 📂 Sources:")
            shown = set()
            for doc in result['source_documents']:
                src = doc.metadata['source']
                if src not in shown:
                    with st.expander(f"{src}"):
                        st.markdown(f"```\n{doc.page_content.strip()[:1500]}\n```)  # full text preview")
                    shown.add(src)
else:
    st.info("⬅️ Upload files from the sidebar to get started")
