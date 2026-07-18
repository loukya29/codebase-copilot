"""
tools.py
--------
LangChain tool definitions the agent can call. Each tool is a thin,
well-scoped wrapper around a real subprocess or the RAG retriever --
the agent decides WHEN to call them, these just execute reliably.

Safety note: run_tests / run_linter execute subprocess commands against
the target repo. For anything beyond your own sample/local repos, run
this inside a container or VM with resource limits -- don't point it at
an untrusted repo on your host machine as-is.
"""

import subprocess
from pathlib import Path

from langchain_core.tools import tool

from retriever import CodebaseRetriever

# Module-level retriever instance, set up once by agent.py via set_repo_path()
_retriever: CodebaseRetriever | None = None
_repo_path: str | None = None


def set_repo_path(repo_path: str, retriever: CodebaseRetriever):
    """Wire up the tools to point at a specific repo + pre-built retriever."""
    global _retriever, _repo_path
    _retriever = retriever
    _repo_path = repo_path


@tool
def search_codebase(query: str) -> str:
    """Search the codebase for functions/classes relevant to a natural-language
    query, e.g. 'where is division handled' or 'user authentication logic'.
    Returns the most relevant code chunks with file names and line numbers.
    Use this before proposing a fix, so you ground your answer in real code."""
    if _retriever is None:
        return "Error: no repo has been indexed yet."
    results = _retriever.search(query, k=4)
    if not results:
        return "No relevant code found."
    formatted = []
    for r in results:
        formatted.append(f"--- {r['file']}::{r['name']} (lines {r['lines']}) ---\n{r['content']}")
    return "\n\n".join(formatted)


@tool
def read_file(file_path: str) -> str:
    """Read the full contents of a specific file in the repo, given a path
    relative to the repo root. Use this when you need more context than a
    single retrieved chunk gives you, e.g. to see imports or surrounding code."""
    if _repo_path is None:
        return "Error: no repo path set."
    full_path = Path(_repo_path) / file_path
    if not full_path.exists():
        return f"Error: {file_path} not found in repo."
    try:
        return full_path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def run_tests() -> str:
    """Run the repo's pytest test suite and return the results, including
    any failures and tracebacks. Use this to check whether the code currently
    works, or to verify whether a proposed fix would resolve a failure."""
    if _repo_path is None:
        return "Error: no repo path set."
    try:
        result = subprocess.run(
            ["python3", "-m", "pytest", "-v", "--tb=short"],
            cwd=_repo_path,
            capture_output=True,
            text=True,
            timeout=60,
        )
        output = result.stdout + "\n" + result.stderr
        # Trim overly long output so it doesn't blow the agent's context window
        return output[-4000:] if len(output) > 4000 else output
    except subprocess.TimeoutExpired:
        return "Error: test run timed out after 60 seconds."
    except Exception as e:
        return f"Error running tests: {e}"


@tool
def run_linter() -> str:
    """Run a linter (ruff) over the repo and return style/quality issues found.
    Use this to check for problems beyond what tests catch, like unused
    imports, undefined names, or style violations."""
    if _repo_path is None:
        return "Error: no repo path set."
    try:
        result = subprocess.run(
            ["ruff", "check", "."],
            cwd=_repo_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + "\n" + result.stderr
        return output if output.strip() else "No linting issues found."
    except FileNotFoundError:
        return "Error: ruff is not installed. Run: pip install ruff"
    except subprocess.TimeoutExpired:
        return "Error: linter run timed out."
    except Exception as e:
        return f"Error running linter: {e}"


@tool
def git_diff() -> str:
    """Return the current uncommitted git diff for the repo, if it's a git
    repository. Use this to understand what changed recently before
    summarizing a PR or explaining recent modifications."""
    if _repo_path is None:
        return "Error: no repo path set."
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=_repo_path,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return f"Not a git repo, or git error: {result.stderr}"
        return result.stdout if result.stdout.strip() else "No uncommitted changes."
    except FileNotFoundError:
        return "Error: git is not installed."
    except Exception as e:
        return f"Error getting git diff: {e}"


ALL_TOOLS = [search_codebase, read_file, run_tests, run_linter, git_diff]
