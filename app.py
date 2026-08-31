"""
app.py
------
Gradio UI for Codebase Copilot. Point it at a local repo path, then chat
with the agent about that codebase.

Run:
    python app.py
Then open the local URL Gradio prints.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gradio as gr
from agent import build_agent, chat
from dotenv import load_dotenv
load_dotenv()

state = {"agent": None, "repo": None}


def index_repo(repo_path: str) -> str:
    if not repo_path or not os.path.isdir(repo_path):
        return "⚠️ That path doesn't exist. Enter a valid local repo path."
    try:
        agent, n = build_agent(repo_path)
        state["agent"] = agent
        state["repo"] = repo_path
        return f"✅ Indexed {n} code chunks from `{repo_path}`. Ask away below."
    except Exception as e:
        return f"⚠️ Failed to index repo: {e}"


def respond(message, history):
    if state["agent"] is None:
        return "Please index a repo first (top of the page) before chatting."
    try:
        return chat(state["agent"], message)
    except Exception as e:
        return f"Error from agent: {e}"


with gr.Blocks(title="Codebase Copilot") as demo:
    gr.Markdown("# 🧭 Codebase Copilot\nRAG + agentic debugging assistant for a codebase.")

    with gr.Row():
        repo_input = gr.Textbox(
            label="Repo path (local folder)",
            placeholder="e.g. ./sample_repo",
            value="./sample_repo",
        )
        index_btn = gr.Button("Index repo", variant="primary")

    index_status = gr.Markdown()
    index_btn.click(index_repo, inputs=repo_input, outputs=index_status)

    gr.ChatInterface(
        fn=respond,
        examples=[
            "What does the average function do?",
            "Run the tests and tell me what's failing.",
            "Find the bug causing the failing test and suggest a fix.",
            "Are there any linter issues?",
        ],
    )

if __name__ == "__main__":
    demo.launch()
