"""
agent.py
--------
Wires up the tool-calling agent: give it a repo, it indexes the repo for
RAG, then answers questions and debugs using search_codebase, read_file,
run_tests, run_linter, and git_diff as tools.

Model choice note (worth knowing for interviews): small open-source HF
models are often unreliable at *tool calling* specifically -- they need
strong instruction-following + function-calling training that most small
models don't have. So this project splits the two jobs:
  - Hugging Face model -> embeddings / retrieval (runs fully locally, free)
  - A strong tool-calling model -> agent orchestration (Anthropic/OpenAI,
    or a HF model that explicitly supports tool calling, e.g. via HF
    Inference Providers with something like Qwen2.5-72B-Instruct)

"""

import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from retriever import CodebaseRetriever
import tools as tools_module

from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = """You are Codebase Copilot, an agent that helps developers \
understand, test, and debug a specific codebase.

You have tools to search the codebase semantically, read full files, run \
the test suite, run a linter, and check the current git diff.

Rules:
- Always ground your answers in the actual code -- use search_codebase or \
read_file before making claims about what the code does.
- If asked to debug or fix something, first run_tests to see the actual \
failure, then search_codebase / read_file to find the relevant function, \
then propose a specific, minimal fix with your reasoning.
- When proposing a fix, show the exact code change, don't just describe it \
in prose.
- If you're not sure a fix is correct, say so, and suggest running the \
tests again to confirm.
- IMPORTANT: You do NOT have the ability to write or modify files. You can \
only propose changes as text/code blocks. If asked to "apply" a fix, clarify \
that you can only suggest the change -- the user must apply it themselves -- \
and never claim you've modified a file when you haven't."""


def build_agent(repo_path: str, model_provider: str | None = None):
    """
    Index a repo and build the tool-calling agent pointed at it.

    Args:
        repo_path: path to the repo to index and debug
        model_provider: e.g. "anthropic:claude-sonnet-4-5" or "openai:gpt-4o-mini".
            Defaults to the MODEL_PROVIDER env var.

    Returns:
        (agent, chunk_count) -- the compiled agent graph and how many
        code chunks were indexed
    """
    model_provider = model_provider or os.environ.get(
        "MODEL_PROVIDER", "openai:gpt-4o-mini"
    )

    retriever = CodebaseRetriever()
    chunk_count = retriever.build_index(repo_path)
    tools_module.set_repo_path(repo_path, retriever)

    model = init_chat_model(model_provider)
    agent = create_agent(
        model=model,
        tools=tools_module.ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        debug=True,
    )
    return agent, chunk_count


def chat(agent, message: str) -> str:
    """Send one message to the agent and return its final text response."""
    result = agent.invoke({"messages": [{"role": "user", "content": message}]})
    final_message = result["messages"][-1]
    return final_message.content


if __name__ == "__main__":
    import sys

    repo = sys.argv[1] if len(sys.argv) > 1 else "../sample_repo"
    print(f"Indexing {repo} ...")
    agent, n = build_agent(repo)
    print(f"Indexed {n} chunks. Agent ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break
        response = chat(agent, user_input)
        print(f"\nAgent: {response}\n")
