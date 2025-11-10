from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import streamlit as st
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import re

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

st.cache_data.clear()

pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=200,
    temperature=0.5,
)
llm = HuggingFacePipeline(pipeline=pipe)

def n_tokens(text: str) -> int:
    return len(tokenizer(text)["input_ids"])

@st.cache_resource
def load_pdf():
    pdf_name = "Metab_Serotonin.pdf"
    loader = UnstructuredPDFLoader(pdf_name, mode="elements")
    docs = loader.load()

    joined_docs = []
    for d in docs:
        joined_docs.append(d)

    all_text = "\n\n".join([d.page_content for d in joined_docs])

    refs_match = re.search(r'\n\s*(References|Bibliography|REFERENCES|BIBLIOGRAPHY)\s*\n', all_text)
    if refs_match:
        all_text = all_text[: refs_match.start()]

    from langchain.schema import Document
    cleaned_doc = Document(page_content=all_text, metadata={})
    docs = [cleaned_doc]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", "? ", "! ", ", ", " "],
    )
    chunks = text_splitter.split_documents(docs)

    def clean_text(s: str) -> str:
        s = s.replace("\xa0", " ")
        s = re.sub(r'([a-zA-Z])\d+(?:,\d+)*', r'\1', s)
        s = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', s)
        s = re.sub(r'\(\s*\d+(?:\s*,\s*\d+)*\s*\)', '', s)
        s = re.sub(r'^\s*\d{1,3}\.\s+[A-Z].*$', '', s, flags=re.MULTILINE)
        s = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def strip_references(text: str) -> str:
        text = re.split(r'\bReferences\b|\bREFERENCES\b', text)[0]
        text = re.sub(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]', '', text)
        text = re.sub(r'\(\s*\d+(?:\s*,\s*\d+)*\s*\)', '', text)
        text = re.sub(r'\b\d+\s*\.\s+[A-Z][a-z]', '', text)
        return text.strip()

    def is_junk(s: str) -> bool:
        s_strip = s.strip()

        if len(s_strip) < 50:
            return True

        words = re.findall(r'\w+', s_strip)
        if len(words) < 5:
            return True
        return False

    cleaned_chunks = []
    for d in chunks:
        txt = clean_text(d.page_content)
        txt = strip_references(txt)
        if not is_junk(txt):
            d.page_content = txt
            cleaned_chunks.append(d)

    cleaned_chunks = [
        d for d in cleaned_chunks
        if "references" not in d.page_content.lower()
    ]

    merged = []
    i = 0
    while i < len(cleaned_chunks):
        cur = cleaned_chunks[i].page_content
        j = i + 1
        while len(cur) < 500 and j < min(i + 4, len(cleaned_chunks)):
            cur += " " + cleaned_chunks[j].page_content
            j += 1
        doc = cleaned_chunks[i]
        doc.page_content = cur
        merged.append(doc)
        i = j

    final_chunks = []
    max_tokens = 400
    for d in merged:
        if n_tokens(d.page_content) <= max_tokens:
            final_chunks.append(d)
        else:
            parts = re.split(r'(?<=[.?!])\s+', d.page_content)
            buffer = ""
            for p in parts:
                if n_tokens(buffer + " " + p) <= max_tokens:
                    buffer = (buffer + " " + p).strip()
                else:
                    if buffer:
                        doc = d
                        doc.page_content = buffer
                        final_chunks.append(doc)
                    buffer = p
            if buffer:
                doc = d
                doc.page_content = buffer
                final_chunks.append(doc)

    emb_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = FAISS.from_documents(final_chunks, emb_model)
    return vectorstore

vectorstore = load_pdf()

template = """You are an academic assistant in the field of metabolomics. 
Use ONLY the context below to answer the question clearly and concisely.
Respond naturally, add contextual explanations as needed.
Use correct grammar and punctuation.
Do not include citation numbers or reference lists in your answer.
If the answer is not contained in the context, reply with "I don't know".

Context:
{context}

Question:
{question}

Answer:"""

qa_prompt = PromptTemplate(template=template, input_variables=["context", "question"])

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": qa_prompt},
)

docs = vectorstore.similarity_search("serotonin and alzheimers", k=10)
for i, d in enumerate(docs, 1):
    print(f"\n---- Chunk #{i} ----")
    print(d.page_content[:500])

st.title("Academic Paper QA System")
prompt = st.chat_input("Enter question")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    answer = chain.invoke({"query": prompt})

    st.chat_message("assistant").markdown(answer["result"])
    st.session_state.messages.append({"role": "assistant", "content": answer["result"]})