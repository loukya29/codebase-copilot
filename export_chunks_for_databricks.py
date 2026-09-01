"""
export_chunks_for_databricks.py
---------------------------------
Runs your existing AST-based chunker and exports the results as a single
JSON file, ready to upload into a Databricks workspace (via the UI upload,
or a Unity Catalog Volume) and load into a Delta table.

This deliberately does NOT require any Databricks SDK or network access --
it just reuses ingest.py, which you already have working, and writes
plain JSON. Keeping this step separate from the Databricks side means you
can regenerate this file anytime locally, with zero risk of your event-day
Databricks setup blocking your ability to produce fresh data.

Run:
    python export_chunks_for_databricks.py sample_repo chunks_export.json
"""

import sys
import json
from datetime import datetime, timezone
sys.path.insert(0, "src")

from ingest import chunk_repo


def main():
    repo_path = sys.argv[1] if len(sys.argv) > 1 else "sample_repo"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "chunks_export.json"

    chunks = chunk_repo(repo_path)

    # Add a couple of fields useful once this lives in a table --
    # when it was indexed, and code length, for quick SQL filtering later
    # (e.g. "show me all functions over 500 characters", tying back to
    # your chunking-strategy comparison work).
    indexed_at = datetime.now(timezone.utc).isoformat()
    for c in chunks:
        c["indexed_at"] = indexed_at
        c["code_length"] = len(c["code"])
        c["has_docstring"] = c.get("docstring") is not None

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    print(f"Exported {len(chunks)} chunks from '{repo_path}' to '{output_path}'")
    print(f"File size: {len(json.dumps(chunks))} bytes")
    print("\nUpload this file into your Databricks workspace next "
          "(Catalog Explorer -> your Volume -> Upload, or the "
          "workspace file upload UI).")


if __name__ == "__main__":
    main()