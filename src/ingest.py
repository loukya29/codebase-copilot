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
    start = node.lineno - 1 # to match python indexing
    end = node.end_lineno  # end_lineno is inclusive, slicing is exclusive -> correct
    return "\n".join(source_lines[start:end]) #slices out just those lines, "\n".join(...) glues them back into one string with newlines restored.


def _chunk_file(file_path: Path, repo_root: Path) -> List[Dict]:
    """Parse a single Python file into function/class-level chunks."""
    chunks = []
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return chunks

    try:
        tree = ast.parse(source)
        #ast.parse() gives you a tree of nodes representing your code's structure. it carries only the line number not the source text

        #So to actually get the text of a function, you have to go back to the original file's lines and slice out the right range yourself.
    except SyntaxError:
        # Skip files that don't parse (e.g. Python 2 syntax, templates, etc.)
        return chunks

    source_lines = source.splitlines()
    rel_path = str(file_path.relative_to(repo_root))#this turns /Users/loukya/.../sample_repo/calculator.py into just calculator.py

    # Module-level docstring as its own chunk (useful for "what does this file do")
    module_doc = ast.get_docstring(tree) #Before walking individual functions, grab the file-level docstring if one exists (the triple-quoted string at the very top of a file which explains what each file does)
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

    for node in ast.walk(tree): #ast.walk(tree) - visits every node in the tree recursively including all functions, classes and variable assignment and every if loop. But this becomes too much.
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): # this takes only the classes functions and async functions. if it takes if loop and all, then it will be out of context
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

    #Converts the string path into a Path object and resolves it to an absolute path (e.g. turns "../sample_repo" into /Users/loukya/Desktop/codebase-copilot/sample_repo)
    repo_root = Path(repo_path).resolve()
    all_chunks = []


    #os.walk = recursive engine
    #walks the entire folder tree, starting at repo_root. and for every folder it visits it going to return 3 things -
    # 1. current folders path - dirpath
    # 2. the list of subfolder names - dirnames
    # 3. list of filenames inside the subfolder - filename
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for filename in filenames:
            if filename.endswith(".py"):
                file_path = Path(dirpath) / filename #building the entire file path
                # pass all the python files from the directory into the _chunk_file function
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
