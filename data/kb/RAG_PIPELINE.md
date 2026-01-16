# RAG Pipeline (Local, Grounded)

This repository includes a local Retrieval-Augmented Generation (RAG) copilot that answers questions using ONLY the project’s documentation and reports.

## Ingestion
- Input: files under `data/kb/` (Markdown, text, optional PDFs)
- Chunking: `RecursiveCharacterTextSplitter` with configurable chunk size/overlap
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: ChromaDB persisted under `artifacts/chroma`
- Collection name: `finance_copilot`

Each chunk stores metadata:
- `source`: original file path
- `chunk_id`: stable per-file chunk identifier
- `page`: PDF page if applicable
- `ticker`: ticker tag (e.g., AAPL)

## Retrieval
At query time, the copilot:
1) Loads the persisted Chroma collection
2) Runs similarity search (top-k)
3) Returns retrieved chunks + metadata

## Answering (Strict Grounding)
The LLM is instructed to:
- Use ONLY retrieved context
- If missing, respond exactly: `Not found in docs.`
- Provide a `Citations:` line listing chunk_ids used

If the answer does not include valid chunk citations, the system returns `Not found in docs.` instead of guessing.

## Outputs
The API returns:
- `answer`: grounded response or `Not found in docs.`
- `citations`: list of sources, chunk_ids, snippets
- `retrieval_debug`: vector-store counts, top sources, scores
