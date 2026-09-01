"""
retriever_databricks.py
-------------------------
A Databricks Vector Search-backed retriever, deliberately matching the
same interface as CodebaseRetriever in retriever.py:
    - .search(query, k) -> List[Dict] with the same keys
      (file, name, type, lines, content)

Because tools.py's search_codebase() only ever calls `_retriever.search(...)`
and reads that exact dict shape, it can use EITHER retriever without any
changes -- swap which class gets instantiated in agent.py's build_agent(),
and everything downstream (the agent, the tool, the system prompt) keeps
working unmodified. This is the actual point of having isolated the
retriever behind one interface from the start.

Prerequisites (done via databricks_notebook_delta_ingest.py first):
    1. Chunks loaded into a Delta table
    2. A Vector Search endpoint created
    3. A Delta Sync Index created on that table

Auth: set these environment variables (or rely on Databricks CLI profile
config if running inside a Databricks notebook/job, where auth is automatic):
    DATABRICKS_HOST   e.g. https://<your-workspace>.cloud.databricks.com
    DATABRICKS_TOKEN  a personal access token
"""

import os
from typing import List, Dict, Optional

from databricks.vector_search.client import VectorSearchClient


class DatabricksCodebaseRetriever:
    def __init__(
        self,
        endpoint_name: str,
        index_name: str,
        databricks_host: Optional[str] = None,
        databricks_token: Optional[str] = None,
    ):
        """
        Args:
            endpoint_name: the Vector Search endpoint created in the notebook
                (e.g. "codebase_copilot_vs_endpoint")
            index_name: fully-qualified index name, catalog.schema.index
                (e.g. "workspace.default.codebase_chunks_index")
            databricks_host / databricks_token: credentials. If omitted,
                falls back to DATABRICKS_HOST / DATABRICKS_TOKEN env vars,
                or ambient auth if running inside a Databricks notebook.
        """
        self.endpoint_name = endpoint_name
        self.index_name = index_name

        client_kwargs = {}
        host = databricks_host or os.environ.get("DATABRICKS_HOST")
        token = databricks_token or os.environ.get("DATABRICKS_TOKEN")
        if host and token:
            client_kwargs = {"workspace_url": host, "personal_access_token": token}

        self.client = VectorSearchClient(**client_kwargs)
        self.index = self.client.get_index(
            endpoint_name=endpoint_name, index_name=index_name
        )

    def build_index(self, repo_path: str) -> int:
        """
        No-op for interface compatibility with CodebaseRetriever.

        Unlike the local FAISS path, indexing here happens in the Delta
        table + Vector Search pipeline (databricks_notebook_delta_ingest.py),
        NOT in this Python process. This method exists only so agent.py's
        build_agent() can call it identically regardless of which retriever
        backend is active, without an if/else branch at the call site.
        """
        return -1  # signals "chunk count not tracked locally for this backend"

    def search(self, query: str, k: int = 4) -> List[Dict]:
        """Same return shape as CodebaseRetriever.search() in retriever.py --
        this is what makes the swap transparent to tools.py."""
        raw_results = self.index.similarity_search(
            query_text=query,
            columns=["id", "file", "name", "type", "start_line", "end_line", "code"],
            num_results=k,
        )

        # Databricks Vector Search returns results under
        # raw_results["result"]["data_array"], each row matching the
        # `columns` order requested above, with a trailing similarity score.
        rows = raw_results.get("result", {}).get("data_array", [])
        columns = ["id", "file", "name", "type", "start_line", "end_line", "code"]

        formatted = []
        for row in rows:
            record = dict(zip(columns, row))
            formatted.append({
                "file": record["file"],
                "name": record["name"],
                "type": record["type"],
                "lines": f"{record['start_line']}-{record['end_line']}",
                "content": record["code"],
            })
        return formatted


if __name__ == "__main__":
    import sys

    endpoint = sys.argv[1] if len(sys.argv) > 1 else "codebase_copilot_vs_endpoint"
    index_name = sys.argv[2] if len(sys.argv) > 2 else "workspace.default.codebase_chunks_index"
    query = sys.argv[3] if len(sys.argv) > 3 else "where is division handled"

    retriever = DatabricksCodebaseRetriever(endpoint_name=endpoint, index_name=index_name)
    results = retriever.search(query, k=4)
    for r in results:
        print(f"{r['file']}::{r['name']} (lines {r['lines']})")