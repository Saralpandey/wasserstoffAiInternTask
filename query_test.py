from vector_db import query_chroma

# STEP 1: Write a query that matches actual stored content
query = "Bunks Prediction Model"

# STEP 2: Fetch results from vector DB
results = query_chroma(query)

# STEP 3: Print matched docs
if not results['documents'][0]:
    print("❌ No relevant documents found. Try changing the query.")
else:
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        print(f"📄 From: {meta['source']}")
        print(doc[:400])  # Show first 400 characters of matched doc
        print("─" * 60)
