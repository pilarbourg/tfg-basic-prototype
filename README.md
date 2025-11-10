# Academic Paper QA Prototypes

This repository contains early experiments for building a question-answering system over academic papers. Each prototype demonstrates a slightly different approach for ingesting PDFs, creating embeddings, and querying the content.

---

## Prototype 1 – Single Paper QA
- **Purpose:** Quickly query a single PDF.
- **How it works:**  
  1. Load the PDF and split the text into fixed-size chunks.  
  2. Encode each chunk using `SentenceTransformer`.  
  3. Store embeddings in a FAISS index.  
  4. Run a simple CLI where the user can type a question and retrieve the top 3 most similar chunks.
- **Limitations:** Only works on one paper at a time; no handling of multiple labels or multiple documents.

---

## Prototype 2 – Multi-Paper QA with Labels
- **Purpose:** Query multiple PDFs and track labels.  
- **How it works:**  
  1. Accept multiple PDFs with a simple label for each (e.g., positive/negative).  
  2. Chunk each PDF and generate embeddings for all chunks.  
  3. Store embeddings in FAISS.  
  4. CLI allows querying across all papers, returning top 5 results with similarity scores and labels.  
- **Differences from Prototype 1:** Supports multiple papers and tracks additional metadata like labels and approximate pages.

---

## Prototype 3 – Large-Scale Multi-Paper QA
- **Purpose:** Handle many papers efficiently with normalized embeddings.  
- **How it works:**  
  1. Load a list of PDFs and split them into paragraphs, skipping initial headers.  
  2. Combine paragraphs into chunks of roughly 500 words.  
  3. Encode chunks with `SentenceTransformer`, normalize embeddings, and store in FAISS using inner product similarity.  
  4. CLI allows querying and ranks results by similarity with simple relevance categories (high, moderate, low).  
- **Differences from Prototype 2:** Scales to more papers, uses normalized embeddings, and provides relevance ranking.

---

### Notes
- All prototypes use FAISS for fast vector search and `SentenceTransformer` embeddings.  
- Prototypes are meant as a starting point; more sophisticated pipelines (e.g., LLM-based QA, chunk cleaning, citation handling) can be built on top.  
- These scripts are experimental and designed for CLI-based exploration.