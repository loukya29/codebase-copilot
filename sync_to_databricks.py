"""
sync_to_databricks.py
-----------------------
Fully automates re-indexing after your local codebase changes, with ZERO
manual steps (no export-then-upload-then-run-notebook-cells dance):

    1. Re-chunks the repo locally (same ingest.py you already have)
    2. Computes embeddings locally (same sentence-transformers model)
    3. Writes directly into the Delta table over a SQL connection --
       no file upload, no Volume, no notebook
    4. Triggers a Vector Search index sync via the API

Run this any time your repo changes and you want the Databricks-backed
retriever to reflect the update:

    python sync_to_databricks.py sample_repo

Requires these environment variables (put them in your .env, same file
your OpenAI/Anthropic key already lives in):

    DATABRICKS_HOST         e.g. https://dbc-xxxx.cloud.databricks.com
    DATABRICKS_TOKEN        a personal access token
    DATABRICKS_HTTP_PATH    the SQL Warehouse's HTTP path (see note below)

Getting DATABRICKS_HTTP_PATH: in the Databricks UI, go to SQL Warehouses,
click your warehouse, "Connection Details" tab -- copy the "HTTP path"
value shown there. Free Edition includes a default serverless SQL
warehouse you can use for this.
"""

import sys
import os
sys.path.insert(0, "src")

from dotenv import load_dotenv
load_dotenv()

from ingest import chunk_repo
from sentence_transformers import SentenceTransformer
from databricks import sql
from databricks.vector_search.client import VectorSearchClient

CATALOG = "workspace"
SCHEMA = "default"
TABLE = "codebase_chunks"
ENDPOINT_NAME = "codebase_copilot_vs_endpoint"
INDEX_NAME = "codebase_chunks_index"
FULL_TABLE = f"{CATALOG}.{SCHEMA}.{TABLE}"
FULL_INDEX = f"{CATALOG}.{SCHEMA}.{INDEX_NAME}"


def build_text_for_embedding(chunk: dict) -> str:
    """Same docstring+code concatenation trick used everywhere else in
    this project -- kept consistent across FAISS, the notebook, and here."""
    parts = [f"# {chunk['file']}::{chunk['name']} ({chunk['type']})"]
    if chunk.get("docstring"):
        parts.append(chunk["docstring"])
    parts.append(chunk["code"])
    return "\n".join(parts)


def main():
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "sample_repo"

    host = os.environ["DATABRICKS_HOST"]
    token = os.environ["DATABRICKS_TOKEN"]
    http_path = os.environ["DATABRICKS_HTTP_PATH"]

    # 1. Re-chunk locally
    print(f"Chunking {repo_path}...")
    chunks = chunk_repo(repo_path)
    print(f"  {len(chunks)} chunks found")

    # 2. Embed locally
    print("Computing embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = [build_text_for_embedding(c) for c in chunks]
    embeddings = model.encode(texts).tolist()

    # 3. Write directly to the Delta table over SQL -- no file upload needed
    print(f"Writing {len(chunks)} rows to {FULL_TABLE}...")
    with sql.connect(
        server_hostname=host.replace("https://", ""),
        http_path=http_path,
        access_token=token,
    ) as conn:
        with conn.cursor() as cursor:
            # Rebuild the table fresh each sync -- simplest correct approach
            # for a project at this scale. A real production version would
            # MERGE instead of overwrite, to avoid a brief empty-table gap.
            cursor.execute(f"DELETE FROM {FULL_TABLE}")

            for chunk, text, emb in zip(chunks, texts, embeddings):
                cursor.execute(
                    f"""
                    INSERT INTO {FULL_TABLE}
                    (id, file, name, type, start_line, end_line, docstring,
                     code, text_for_embedding, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        chunk["id"], chunk["file"], chunk["name"], chunk["type"],
                        chunk["start_line"], chunk["end_line"],
                        chunk.get("docstring"), chunk["code"], text, emb,
                    ],
                )
    print("  Write complete")

    # 4. Trigger the Vector Search index to sync with the updated table
    print("Triggering Vector Search sync...")
    vs_client = VectorSearchClient(
        workspace_url=host, personal_access_token=token
    )
    index = vs_client.get_index(endpoint_name=ENDPOINT_NAME, index_name=FULL_INDEX)
    index.sync()
    print("  Sync triggered -- index will update in the background")

    print(f"\nDone. {len(chunks)} chunks re-indexed in Databricks.")


if __name__ == "__main__":
    main()