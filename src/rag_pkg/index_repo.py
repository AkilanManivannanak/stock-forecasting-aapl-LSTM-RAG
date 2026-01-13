from pathlib import Path
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


EXCLUDE_DIRS = {".git", "__pycache__", "venv", ".venv", "outputs", "models"}
EXCLUDE_EXTS = {".jpg", ".png", ".ipynb", ".csv", ".pkl", ".h5", ".dmg"}


def iter_repo_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        if not p.is_file():
            continue
        if p.suffix in EXCLUDE_EXTS:
            continue
        if p.suffix in {".py", ".md", ".txt"}:
            files.append(p)
    return files


def load_file_with_line_chunks(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    chunk_size = 80
    overlap = 20

    docs: List[Document] = []
    start = 0
    while start < len(lines):
        end = min(len(lines), start + chunk_size)
        chunk_lines = lines[start:end]
        content = "\n".join(
            f"{i+1}: {line}" for i, line in enumerate(chunk_lines, start=start)
        )
        docs.append(
            Document(
                page_content=content,
                metadata={
                    "source": str(path),
                    "start_line": start + 1,
                    "end_line": end,
                },
            )
        )
        if end == len(lines):
            break
        start = end - overlap
    return docs


def build_index(repo_root: str = "."):
    root = Path(repo_root)
    docs: List[Document] = []
    for f in iter_repo_files(root):
        docs.extend(load_file_with_line_chunks(f))

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory="outputs/rag_chroma",
    )


if __name__ == "__main__":
    build_index()
