import argparse
import hashlib
import os
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings


SUPPORTED_TEXT_EXTS = {".txt", ".md"}
SUPPORTED_PDF_EXTS = {".pdf"}
COLLECTION_NAME_DEFAULT = "finance_copilot"


def iter_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (SUPPORTED_TEXT_EXTS | SUPPORTED_PDF_EXTS):
            files.append(p)
    return sorted(files)


def load_docs(path: Path):
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_PDF_EXTS:
        docs = PyPDFLoader(str(path)).load()
        for d in docs:
            d.metadata.setdefault("source", str(path))
        return docs

    docs = TextLoader(str(path), encoding="utf-8").load()
    for d in docs:
        d.metadata.setdefault("source", str(path))
    return docs


def _stable_id(source: str, ticker: str, chunk_id: str, page: Optional[int]) -> str:
    raw = f"{ticker}|{source}|{chunk_id}|{page}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", default=os.getenv("RAG_DOCS_DIR", "data/kb"))
    parser.add_argument("--persist_dir", default=os.getenv("RAG_PERSIST_DIR", "artifacts/chroma"))
    parser.add_argument("--collection_name", default=os.getenv("RAG_COLLECTION_NAME", COLLECTION_NAME_DEFAULT))
    parser.add_argument("--chunk_size", type=int, default=900)
    parser.add_argument("--chunk_overlap", type=int, default=150)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument(
        "--hf_embed_model",
        default=os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
    )
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir)
    persist_dir = Path(args.persist_dir)

    if not docs_dir.exists():
        raise SystemExit(f"Docs dir not found: {docs_dir}")

    persist_dir.mkdir(parents=True, exist_ok=True)

    files = iter_files(docs_dir)
    if not files:
        raise SystemExit(f"No supported docs found in {docs_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    embeddings = HuggingFaceEmbeddings(model_name=args.hf_embed_model)

    vs = Chroma(
        collection_name=args.collection_name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
    )

    all_chunks = []
    all_ids = []

    for f in files:
        docs = load_docs(f)
        for d in docs:
            d.metadata["ticker"] = args.ticker.upper()
            d.metadata.setdefault("source", str(f))

        chunks = splitter.split_documents(docs)
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = f"{str(f)}:{i}"
            page = c.metadata.get("page", None)
            cid = _stable_id(
                source=c.metadata.get("source", str(f)),
                ticker=args.ticker.upper(),
                chunk_id=c.metadata["chunk_id"],
                page=page,
            )
            all_chunks.append(c)
            all_ids.append(cid)

    vs.add_documents(all_chunks, ids=all_ids)

    try:
        vs.persist()
    except Exception:
        pass
    try:
        vs._client.persist()
    except Exception:
        pass

    count = vs._collection.count()

    print(f"[INGEST] Files: {len(files)}")
    print(f"[INGEST] Chunks indexed: {len(all_chunks)}")
    print(f"[INGEST] Persist dir: {persist_dir}")
    print(f"[INGEST] Collection: {args.collection_name}")
    print(f"[INGEST] Embeddings: {args.hf_embed_model}")
    print(f"[INGEST] Chroma count: {count}")
    print("[INGEST] Done.")


if __name__ == "__main__":
    main()
