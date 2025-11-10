from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap
import fitz
import os

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

pdf_files = [
    "Trypto_Metab_CNS.pdf",
    "Renal_Cyst_Fluid.pdf",
    "Insights_Epidemiological_Studies.pdf",
    "Kynu_Neurodegenerative.pdf",
    "Mech_Action_Brain.pdf",
    "Metab_Serotonin.pdf",
    "Metformin.pdf",
    "Serum_Metab_Profile.pdf",
    "Treatment_GCMS.pdf",
    "Trypto_Urine.pdf"
]

all_chunks = []
metadata = []
chunk_size = 500

for paper in pdf_files:
    doc = fitz.open(paper)
    text = "".join(page.get_text() for page in doc)
    paragraphs = [p for p in text.split("\n") if p.strip() != ""]
    
    paragraphs = paragraphs[5:]

    chunk = ""
    chunk_id = 0
    for para in paragraphs:
        if len(chunk.split()) + len(para.split()) > chunk_size:
            all_chunks.append(chunk)
            metadata.append({"paper": paper, "chunk_id": chunk_id})
            chunk_id += 1
            chunk = para
        else:
            chunk += " " + para
    if chunk:
        all_chunks.append(chunk)
        metadata.append({"paper": paper, "chunk_id": chunk_id})

embeddings = model.encode(all_chunks)
embeddings = np.array(embeddings).astype('float32')

embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("Ask a question or 'exit' to quit\n")

while True:
    query = input("🔍 Enter your query: ")
    if query.lower() == "exit":
        print("Ending script")
        break

    query_emb = model.encode([query]).astype('float32')
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

    D, I = index.search(query_emb, k=5)

    print("\nTop results:\n")
    for score, idx in zip(D[0], I[0]):
        if score > 0.65:
            relevant = "highly relevant"
        elif score > 0.4:
            relevant = "moderately relevant"
        else:
            relevant = "less relevant"
        
        print(f"📄 Paper: {metadata[idx]['paper']}")
        print(f"🔢 Chunk ID: {metadata[idx]['chunk_id']}")
        print(f"🏷️ Relevance: {relevant}")
        print(f"🔗 Similarity score: {score:.4f}")
        print(f"🧩 Text: {all_chunks[idx][:700]}...\n{'-'*80}")


        