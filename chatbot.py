# chatbot_with_groq.py
import os
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter
import chromadb

# ✅ Groq API Key (Free & Fast)
os.environ["GROQ_API_KEY"] = "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"

# 🧠 Load vector DB from existing Chroma store
chroma_client = chromadb.PersistentClient(path="./vector_store")
collection = chroma_client.get_or_create_collection(name="documents")

# 🔍 Retrieve all docs
docs = []
raw = collection.get(include=['documents', 'metadatas'])
for doc, meta in zip(raw['documents'], raw['metadatas']):
    docs.append(Document(page_content=doc, metadata=meta))

# 🧱 Chunk large docs (if needed)
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunked_docs = splitter.split_documents(docs)

# 🔗 Load embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma.from_documents(chunked_docs, embedding, persist_directory="./vector_store")

# 💬 Load LLAMA3 from Groq
llm = ChatGroq(api_key=os.environ["GROQ_API_KEY"], model_name="llama3-8b-8192")

# 🤖 Setup QA chain
qa = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), return_source_documents=True)

# 🚀 Interactive Chat Loop
print("🤖 AI Chatbot Ready! Ask anything about your documents. Type 'exit' to quit.")
while True:
    query = input("\n👤 You: ")
    if query.lower() in ["exit", "quit"]:
        break
    result = qa(query)
    print("\n🤖 AI:", result['result'])
    for doc in result['source_documents']:
        print(f"📄 Source: {doc.metadata['source']}")
    print("-" * 60)
