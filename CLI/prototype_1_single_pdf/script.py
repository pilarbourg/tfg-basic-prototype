from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap
import fitz

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

doc = fitz.open("Trypto_Metab_CNS.pdf")
text = "".join(page.get_text() for page in doc)

chunks = textwrap.wrap(text, 500)

metadata = []
for i, chunk in enumerate(chunks):
    metadata.append({
        "paper": "glutamate_alzheimers.pdf",
        "chunk_id": i,
        "page_estimate": i // 2,
        "label": "positive"
    })

embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings).astype('float32'))

print("\nVector database generated")
print("Type a question about your paper (or 'exit' to quit).\n")

while True:
    query = input("🔍 Enter your query: ")
    if query.lower() == "exit":
        print("Exiting")
        break

    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype('float32'), k=3)

    print("\nTop results:\n")
    for idx in I[0]:
        print(f"Paper: {metadata[idx]['paper']}")
        print(f"Chunk ID: {metadata[idx]['chunk_id']}")
        print(f"Label: {metadata[idx]['label']}")
        print(f"Text: {chunks[idx][:400]}...\n{'-'*80}")