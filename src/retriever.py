"""
retriever.py
------------
Builds and queries a FAISS vector index over code chunks, using a
Hugging Face embedding model. Uses a CODE-specific embedding model
rather than a general sentence-transformer, since general text
embeddings don't capture code semantics (variable names, structure,
control flow) well.

Two embedding options are wired up -- pick based on your hardware:
  - "microsoft/codebert-base"                (heavier, more code-aware)
  - "sentence-transformers/all-MiniLM-L6-v2" (lighter, general-purpose,
                                               good fallback if codebert
                                               is too slow on CPU)
"""

import os
from typing import List, Dict, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from ingest import chunk_repo

DEFAULT_CODE_MODEL = "microsoft/codebert-base"
DEFAULT_FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# chunk_repo gives me plain python dicts. but the langchain vector stores uses a specific class called document. These documents have 2 things
# page_content = the actual content that gets embedded and searched for
# metadata = extra structure data
# CREATES A LANGCHAIN DOCUMENT
def _chunks_to_documents(chunks: List[Dict]) -> List[Document]:
    """Convert raw chunk dicts into LangChain Documents with metadata."""
    docs = []
    for c in chunks:
        # comment-style header that is used to identify the chunk - eg calculator.py::divide (function)
        text_parts = [f"# {c['file']}::{c['name']} ({c['type']})"]
        if c.get("docstring"):
            text_parts.append(f'"""{c["docstring"]}"""')
        text_parts.append(c["code"]) # actual source code
        page_content = "\n".join(text_parts) # Glue the header + docstring + code into one string. this is fed into the embedding model
        #The embedding model only ever "sees" and searches over page_content; it never uses metadata to decide similarity. Metadata is purely for your code to use afterward, to display results usefully or filter them.
        docs.append(Document(
            page_content=page_content,
            metadata={
                "id": c["id"],
                "file": c["file"],
                "name": c["name"],
                "type": c["type"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
            },
        ))
    return docs


class CodebaseRetriever:
    def __init__(self, embedding_model: str = DEFAULT_FALLBACK_MODEL):
        """
        Args:
            embedding_model: HF model id for embeddings. Defaults to the
                lighter general-purpose model so this runs fine on CPU.
                Swap to DEFAULT_CODE_MODEL for better code-specific retrieval
                if you have a GPU or don't mind the extra latency.
        """
        self.embedding_model_name = embedding_model
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore: Optional[FAISS] = None

    def build_index(self, repo_path: str) -> int:
        """Chunk the repo and build a fresh FAISS index. Returns chunk count."""
        chunks = chunk_repo(repo_path)
        if not chunks:
            raise ValueError(f"No indexable Python files found in {repo_path}")

        docs = _chunks_to_documents(chunks)
        #these "docs" are passed into the embedding model"| facebook ai similarity search is the vector stores we are using
        self.vectorstore = FAISS.from_documents(docs, self.embeddings)
        return len(chunks)

    def save(self, index_dir: str):
        if self.vectorstore is None:
            raise RuntimeError("No index built yet -- call build_index() first")
        os.makedirs(index_dir, exist_ok=True)
        self.vectorstore.save_local(index_dir)

    def load(self, index_dir: str):
        self.vectorstore = FAISS.load_local(
            index_dir, self.embeddings, allow_dangerous_deserialization=True
        )

    def search(self, query: str, k: int = 4) -> List[Dict]:
        """Return the top-k most relevant code chunks for a natural-language query."""
        if self.vectorstore is None:
            raise RuntimeError("No index loaded -- call build_index() or load() first")

        results = self.vectorstore.similarity_search(query, k=k)
        return [
            {
                "file": r.metadata["file"],
                "name": r.metadata["name"],
                "type": r.metadata["type"],
                "lines": f"{r.metadata['start_line']}-{r.metadata['end_line']}",
                "content": r.page_content,
            }
            for r in results
        ]


if __name__ == "__main__":
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else "../sample_repo"
    query = sys.argv[2] if len(sys.argv) > 2 else "where do we handle division"

    retriever = CodebaseRetriever()
    n = retriever.build_index(repo)
    print(f"Indexed {n} chunks from {repo}")

    print(f"\nQuery: {query}\n")
    for r in retriever.search(query, k=3):
        print(f"  {r['file']}::{r['name']}  (lines {r['lines']})")
