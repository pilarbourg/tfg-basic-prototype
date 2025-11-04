from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap
import fitz

# 1. Load model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 2. Define your PDFs and their labels
pdf_files = [
    {"path": "Trypto_Metab_CNS.pdf", "label": "positive"},
    {"path": "Renal_Cyst_Fluid.pdf", "label": "negative"}
]

chunks = []
metadata = []

# 3. Loop over PDFs and process
for pdf in pdf_files:
    doc = fitz.open(pdf["path"])
    text = "".join(page.get_text() for page in doc)
    pdf_chunks = textwrap.wrap(text, 500)

    for i, chunk in enumerate(pdf_chunks):
        chunks.append(chunk)
        metadata.append({
            "paper": pdf["path"],
            "chunk_id": i,
            "page_estimate": i // 2,
            "label": pdf["label"]
        })

# 4. Create embeddings for all chunks
embeddings = model.encode(chunks)

# 5. Store in FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype('float32'))

# 6. Interactive Query Loop
print("\n✅ Vector database ready!")
print("Type a question about your paper (or 'exit' to quit).\n")

while True:
    query = input("🔍 Enter your query: ")
    if query.lower() == "exit":
        print("👋 Goodbye!")
        break

    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), k=5)  # top 5 results

    print("\n🔎 Top results:\n")
    for idx, dist in zip(I[0], D[0]):
        print(f"📄 Paper: {metadata[idx]['paper']}")
        print(f"🔢 Chunk ID: {metadata[idx]['chunk_id']}")
        print(f"🏷️ Label: {metadata[idx]['label']}")
        print(f"🔗 Similarity score: {1 / (1 + dist):.4f}")  # higher = more similar
        print(f"🧩 Text: {chunks[idx][:400]}...\n{'-'*80}")