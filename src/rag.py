import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch


DOC_PATHS = [
    "docs/model_report.md",
]

def build_vectorstore_from_docs(doc_paths=DOC_PATHS):
    docs = []
    for path in doc_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Document not found: {path}")
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Local / HF embeddings (no OpenAI)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

def build_rag_chain(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    # Small free text model from Hugging Face Hub
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={"temperature": 0.2, "max_length": 512},
    )

    system_prompt = (
        "You are an expert ML engineer analyzing a stock forecasting project. "
        "Use ONLY the provided context (model_report, metrics, notes) to answer "
        "questions about models, metrics, limitations, and next steps. "
        "If something is not in the context, say you don't know."
    )

    prompt = ChatPromptTemplate.from_template(
        "{system_prompt}\n\nContext:\n{context}\n\nQuestion:\n{question}"
    )

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {
            "system_prompt": RunnablePassthrough(lambda _: system_prompt),
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
    )

    return rag_chain
