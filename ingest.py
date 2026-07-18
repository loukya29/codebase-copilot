"""
ingest.py
---------
Walks a repo, parses each Python file with the `ast` module, and produces
one chunk per top-level function / class / method. This is deliberately
NOT naive fixed-size text splitting -- splitting code by character count
can cut a function in half and wreck retrieval quality. Chunking by
AST node keeps each chunk semantically whole.

Each chunk is a dict:
{
    "id": "path/to/file.py::function_name",
    "file": "path/to/file.py",
    "name": "function_name",
    "type": "function" | "class" | "method",
    "start_line": int,
    "end_line": int,
    "docstring": str | None,
    "code": str,          # full source of the node
}
"""

import ast
import os
from pathlib import Path
from typing import List, Dict, Optional


def _get_source_segment(source_lines: List[str], node: ast.AST) -> str:
    """Reconstruct the exact source text for an AST node using line numbers."""
    start = node.lineno - 1
    end = node.end_lineno  # end_lineno is inclusive, slicing is exclusive -> correct
    return "\n".join(source_lines[start:end])


def _chunk_file(file_path: Path, repo_root: Path) -> List[Dict]:
    """Parse a single Python file into function/class-level chunks."""
    chunks = []
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return chunks

    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Skip files that don't parse (e.g. Python 2 syntax, templates, etc.)
        return chunks

    source_lines = source.splitlines()
    rel_path = str(file_path.relative_to(repo_root))

    # Module-level docstring as its own chunk (useful for "what does this file do")
    module_doc = ast.get_docstring(tree)
    if module_doc:
        chunks.append({
            "id": f"{rel_path}::__module__",
            "file": rel_path,
            "name": "__module__",
            "type": "module_docstring",
            "start_line": 1,
            "end_line": 1,
            "docstring": module_doc,
            "code": module_doc,
        })

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node_type = "class" if isinstance(node, ast.ClassDef) else "function"
            docstring = ast.get_docstring(node)
            code = _get_source_segment(source_lines, node)
            chunks.append({
                "id": f"{rel_path}::{node.name}",
                "file": rel_path,
                "name": node.name,
                "type": node_type,
                "start_line": node.lineno,
                "end_line": node.end_lineno,
                "docstring": docstring,
                "code": code,
            })

    return chunks


def chunk_repo(repo_path: str, ignore_dirs: Optional[List[str]] = None) -> List[Dict]:
    """
    Walk a repo directory and chunk every .py file by function/class.

    Args:
        repo_path: path to the root of the repo to index
        ignore_dirs: directory names to skip (defaults to common noise dirs)

    Returns:
        List of chunk dicts (see module docstring for schema)
    """
    if ignore_dirs is None:
        ignore_dirs = {".git", "__pycache__", "venv", ".venv", "node_modules", ".mypy_cache"}

    repo_root = Path(repo_path).resolve()
    all_chunks = []

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = Path(dirpath) / filename
                all_chunks.extend(_chunk_file(file_path, repo_root))

    return all_chunks


if __name__ == "__main__":
    import json
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else "../sample_repo"
    result = chunk_repo(repo)
    print(f"Indexed {len(result)} chunks from {repo}\n")
    for c in result:
        print(f"  [{c['type']:>16}] {c['id']}  (lines {c['start_line']}-{c['end_line']})")
