from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap
import fitz

# 1. Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 2. Extract text
doc = fitz.open("Trypto_Metab_CNS.pdf")
text = "".join(page.get_text() for page in doc)

# 3. Chunk it
chunks = textwrap.wrap(text, 500)

# 4. Create metadata for each chunk
metadata = []
for i, chunk in enumerate(chunks):
    metadata.append({
        "paper": "glutamate_alzheimers.pdf",
        "chunk_id": i,
        "page_estimate": i // 2,  # rough estimate
        "label": "positive"  # or "negative" later
    })

# 5. Create embeddings
embeddings = model.encode(chunks)

# 6. Store in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype('float32'))

# 7. Interactive Query Loop
print("\n✅ Vector database ready!")
print("Type a question about your paper (or 'exit' to quit).\n")

while True:
    query = input("🔍 Enter your query: ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    # Encode query
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), k=3)

    # Display results
    print("\n🔎 Top results:\n")
    for idx in I[0]:
        print(f"📄 Paper: {metadata[idx]['paper']}")
        print(f"🔢 Chunk ID: {metadata[idx]['chunk_id']}")
        print(f"🏷️ Label: {metadata[idx]['label']}")
        print(f"🧩 Text: {chunks[idx][:400]}...\n{'-'*80}")