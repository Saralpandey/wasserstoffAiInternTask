import chromadb
from chromadb.config import Settings

# ✅ Initialize Chroma Client (latest version)
import chromadb

def initialize_chroma():
    # Now persistent, not in-memory
    client = chromadb.PersistentClient(path="./vector_store")
    return client



# ✅ Save Text to Vector DB
def save_to_chroma(doc_id, text, collection_name="documents"):
    client = initialize_chroma()
    collection = client.get_or_create_collection(name=collection_name)
    
    print(f"\n🧾 DOC PREVIEW: {doc_id} →", text[:300])  # First 300 chars
    
    collection.add(
        documents=[text],
        ids=[doc_id],
        metadatas=[{"source": doc_id}]
    )
    print(f"✅ Saved '{doc_id}' to vector DB.")


# ✅ Search/Query Function
def query_chroma(query_text, collection_name="documents", top_k=3):
    client = initialize_chroma()
    collection = client.get_or_create_collection(name=collection_name)

    # Perform semantic search
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )

    return results
