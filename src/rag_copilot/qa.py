import os
import re
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings


COLLECTION_NAME_DEFAULT = "finance_copilot"


def _load_vs(persist_dir: str, hf_embed_model: str, collection_name: str) -> Chroma:
    emb = HuggingFaceEmbeddings(model_name=hf_embed_model)
    return Chroma(
        collection_name=collection_name,
        embedding_function=emb,
        persist_directory=persist_dir,
    )


def _citations_from_docs(docs: List[Document], scores: Optional[List[float]] = None) -> List[dict]:
    out = []
    scores = scores or [None] * len(docs)
    for d, s in zip(docs, scores):
        out.append(
            {
                "source": d.metadata.get("source"),
                "chunk_id": d.metadata.get("chunk_id"),
                "page": d.metadata.get("page", None),
                "score": s,
                "snippet": (d.page_content or "")[:500].replace("\n", " "),
            }
        )
    return out


def _build_prompt(chunks: List[Tuple[str, str]], question: str) -> str:
    context = "\n\n".join([f"CHUNK_ID: {cid}\n{text}" for cid, text in chunks])

    return f"""You are a strict RAG assistant for a stock-forecasting repository.

Rules:
1) Use ONLY the context below.
2) If the answer is not explicitly in the context, output exactly:
Not found in docs.
3) Do NOT infer, guess, assume, or use general knowledge.
4) If you provide an answer, you MUST include a final line:
Citations: <comma-separated chunk_ids used>

Context:
{context}

Question:
{question}

Answer:
"""


def _call_llm(prompt: str) -> str:
    provider = os.getenv("RAG_PROVIDER", "none").strip().lower()

    if provider == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        model = os.getenv("OLLAMA_MODEL", "llama3.1")
        temperature = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

        try:
            from langchain_ollama import ChatOllama
            llm = ChatOllama(base_url=base_url, model=model, temperature=temperature)
        except Exception:
            from langchain_community.chat_models import ChatOllama
            llm = ChatOllama(base_url=base_url, model=model, temperature=temperature)

        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model, temperature=0)
        resp = llm.invoke(prompt)
        return getattr(resp, "content", str(resp)).strip()

    return "Not found in docs."


def _parse_citations_line(answer: str) -> List[str]:
    m = re.search(r"(?im)^\s*Citations\s*:\s*(.+)\s*$", answer)
    if not m:
        return []
    raw = m.group(1)
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def _strip_citations_line(answer: str) -> str:
    return re.sub(r"(?im)^\s*Citations\s*:\s*.+\s*$", "", answer).strip()


def answer_question(
    question: str,
    ticker: str = "AAPL",
    k: int = 6,
    persist_dir: Optional[str] = None,
    hf_embed_model: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> Dict[str, Any]:
    persist_dir = persist_dir or os.getenv("RAG_PERSIST_DIR", "artifacts/chroma")
    hf_embed_model = hf_embed_model or os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    collection_name = collection_name or os.getenv("RAG_COLLECTION_NAME", COLLECTION_NAME_DEFAULT)

    vs = _load_vs(persist_dir=persist_dir, hf_embed_model=hf_embed_model, collection_name=collection_name)
    count = vs._collection.count()

    results = vs.similarity_search_with_score(question, k=int(k))
    docs = [d for d, _ in results]
    scores = [float(s) for _, s in results]

    citations = _citations_from_docs(docs, scores=scores)

    retrieval_debug = {
        "ticker": ticker.upper(),
        "k": int(k),
        "provider": os.getenv("RAG_PROVIDER", "none"),
        "retrieval_mode": "chroma",
        "collection_name": collection_name,
        "persist_dir": persist_dir,
        "chroma_count": count,
        "docs_returned": len(docs),
        "top_sources": [c.get("source") for c in citations[:3]],
        "top_scores": scores[:3],
    }

    if not docs:
        return {
            "answer": "No context was retrieved from Chroma. Re-run ingest and verify RAG_PERSIST_DIR + collection_name match.",
            "citations": [],
            "retrieval_debug": retrieval_debug,
        }

    chunks: List[Tuple[str, str]] = []
    allowed_chunk_ids = set()
    for d in docs:
        cid = d.metadata.get("chunk_id") or "unknown"
        allowed_chunk_ids.add(cid)
        txt = (d.page_content or "")[:1200]
        chunks.append((cid, txt))

    prompt = _build_prompt(chunks=chunks, question=question)
    raw_answer = _call_llm(prompt)

    cited = _parse_citations_line(raw_answer)
    cleaned = _strip_citations_line(raw_answer)

    if cleaned.strip() == "Not found in docs.":
        return {"answer": "Not found in docs.", "citations": citations, "retrieval_debug": retrieval_debug}

    if not cited:
        return {"answer": "Not found in docs.", "citations": citations, "retrieval_debug": retrieval_debug}

    if any(c not in allowed_chunk_ids for c in cited):
        return {"answer": "Not found in docs.", "citations": citations, "retrieval_debug": retrieval_debug}

    return {"answer": cleaned.strip(), "citations": citations, "retrieval_debug": retrieval_debug}
