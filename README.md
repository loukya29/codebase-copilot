# Codebase Copilot

A RAG + agentic debugging assistant. Point it at a codebase, and it can:
- Answer natural-language questions about what the code does (RAG, grounded in real code via AST-based chunking)
- Run the test suite and report failures
- Run a linter and report issues
- Read specific files for more context
- Check the current git diff
- Propose fixes, grounded in retrieved code and actual test output

## Architecture

```
src/
  ingest.py     - Chunks a repo by function/class using Python's `ast` module
                  (not naive text splitting -- keeps functions whole)
  retriever.py  - Embeds chunks with a Hugging Face model, indexes in FAISS,
                  does semantic search
  tools.py      - LangChain @tool wrappers: search_codebase, read_file,
                  run_tests, run_linter, git_diff
  agent.py      - Builds the tool-calling agent (LangChain create_agent)
                  and wires the tools + retriever together
app.py          - Gradio chat UI
sample_repo/    - A tiny calculator module WITH TWO REAL BUGS, for testing
                  end-to-end before pointing this at a real project
```

## Setup

```bash
pip install -r requirements.txt
```

**Why not a fully open-source model for the agent itself?** Small open
Hugging Face models are generally unreliable at *tool calling* specifically
-- it needs strong function-calling training most small models lack. This
project uses HF models for the part they're genuinely great at (embeddings/
retrieval, fully local and free) and a stronger hosted model for tool-calling
orchestration


## Try these once it's running (against sample_repo)

- "What does the average function do?"
- "Run the tests and tell me what's failing."
- "Find the bug causing the failing test and suggest a fix."

The sample repo has two intentional bugs:
1. `divide()` doesn't handle division by zero
2. `average([])` raises a raw `ZeroDivisionError` instead of a clean `ValueError`

`test_calculator.py::test_average_empty_list` currently fails -- that's
your first real debugging target.

## What's been tested so far

- ✅ `ingest.py` -- AST chunking verified against sample_repo (12 chunks, correct line ranges)
- ✅ `pytest` / `ruff` subprocess calls -- verified working against sample_repo
- ⚠️ `retriever.py` / `agent.py` -- code is complete but embedding downloads
  and LLM calls need network access to huggingface.co and your model
  provider's API, which weren't reachable in the sandbox this was built in.
  Run these yourself and they should work; if you hit an import or API
  error, that's the next thing to debug together.
