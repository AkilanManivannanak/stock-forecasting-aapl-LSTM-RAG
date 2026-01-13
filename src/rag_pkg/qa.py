from typing import List

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI  # or another LLM client


SYSTEM_PROMPT = """You are a repo assistant.
You MUST answer using ONLY the provided context.
If the answer is not clearly in the context, say: "Not found in repo."
Do not guess or invent APIs or files."""


def _load_vs() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory="outputs/rag_chroma",
    )


prompt = ChatPromptTemplate.from_template(
    "{system}\n\nContext:\n{context}\n\nQuestion:\n{question}"
)
parser = StrOutputParser()


def _format_citations(docs: List[Document]) -> str:
    lines = []
    for d in docs:
        src = d.metadata.get("source", "")
        s = d.metadata.get("start_line", "?")
        e = d.metadata.get("end_line", "?")
        lines.append(f"- {src}:L{s}-L{e}")
    return "\n".join(lines)


def answer_repo(question: str) -> str:
    vs = _load_vs()
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    docs: List[Document] = retriever.invoke(question)

    if not docs:
        return "Not found in repo.\n\nSources:\n(none)"

    context = "\n\n".join(d.page_content for d in docs)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm | parser
    answer = chain.invoke(
        {"system": SYSTEM_PROMPT, "context": context, "question": question}
    )

    citations = _format_citations(docs)
    return f"{answer}\n\nSources:\n{citations}"
