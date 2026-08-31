"""
compare_chunking.py
--------------------
Demonstrates the difference between naive fixed-size text chunking and
AST-based chunking on the same file. Run this against sample_repo to see
a long function get cut in half by naive chunking, and stay whole with
AST chunking.

Run:
    python compare_chunking.py sample_repo/calculator.py
"""

import sys
from pathlib import Path
from src.ingest import chunk_repo

sys.path.insert(0, "src")



def naive_chunk(text: str, chunk_size: int = 500):
    """The bad approach: split raw text every N characters, with zero
    awareness of function/class boundaries."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def main():
    file_path = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_repo/calculator.py")
    source = file_path.read_text(encoding="utf-8")

    print("=" * 70)
    print("NAIVE FIXED-SIZE CHUNKING (every 500 characters, no code awareness)")
    print("=" * 70)
    naive_chunks = naive_chunk(source, 500)
    for i, chunk in enumerate(naive_chunks):
        print(f"\n--- naive chunk {i} ({len(chunk)} chars) ---")
        print(chunk)

    print("\n\n")
    print("=" * 70)
    print("AST-BASED CHUNKING (one chunk per function/class)")
    print("=" * 70)
    repo_chunks = chunk_repo(str(file_path.parent))
    for c in repo_chunks:
        if c["file"] == file_path.name and c["type"] == "function":
            print(f"\n--- {c['id']} ({len(c['code'])} chars, lines {c['start_line']}-{c['end_line']}) ---")
            print(c["code"])

    # Specifically call out the long function to make the contrast obvious
    print("\n\n")
    print("=" * 70)
    print("THE POINT: compute_statistics is ", end="")
    target = next((c for c in repo_chunks if c["name"] == "compute_statistics"), None)
    if target:
        print(f"{len(target['code'])} characters long.")
        print("In AST chunking, it's ONE chunk, fully intact, embedded as one coherent unit.")
        print("In naive chunking above, check whether it got split across two 500-char chunks --")
        print("if so, half the function's logic would be embedded separately from the other half,")
        print("meaning a similarity search could match only PART of the function's meaning,")
        print("or retrieve an incomplete, syntactically broken fragment.")
    else:
        print("not found -- did you add compute_statistics to calculator.py?")


if __name__ == "__main__":
    main()