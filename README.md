# Codebase Copilot

A RAG + agentic assistant that explores, tests, and audits a Python codebase — extended with a fine-tuned classifier that flags functions likely to contain bugs. Built as a portfolio project to demonstrate RAG system design, agentic tool use, and rigorous ML evaluation methodology.


## What this project actually does

**Core: an agent that can answer questions about a codebase and act on it.**

Point it at a Python repo and ask questions like "where is division handled" or "find and report all the bugs in this project." Under the hood:

1. The repo is chunked at the function/class level (not fixed-size text blocks) using Python's `ast` module, so each retrievable unit is a whole function or class, not an arbitrary slice of text.
2. Each chunk is embedded (Hugging Face sentence-transformers) and indexed in FAISS for semantic search.
3. A LangChain tool-calling agent is given five tools — `search_codebase`, `read_file`, `run_tests`, `run_linter`, `git_diff` — and a system prompt that instructs it to ground every answer in retrieved code and run tests before proposing a fix. It can find relevant code, read full files, run the test suite, lint, and diff — but it cannot write or apply changes; it reports findings and proposed fixes for a human to act on.

**The actual pitch:** this isn't "another Claude Code / Codex clone" — general-purpose coding agents already do that. The specific use case here is a **pre-assignment code audit tool**: before handing a codebase off to a developer (e.g. for a freelance gig, a new hire's onboarding task, or a hackathon handoff), run one query and get back a report of what's broken, what's untested, and what's risky — *before* work is assigned, not after.

**Extension: a fine-tuned bug-risk classifier.**

On top of the RAG + agent base, I fine-tuned a LoRA adapter on a small transformer to classify a function as "buggy" or "clean," intended as an additional signal for the audit use case (a `score_bug_risk` tool, sketched but not yet wired into the agent — see Limitations). This is the part of the project with the most engineering depth, and it's summarized in detail below because the *process* — not the final accuracy number — is the actual result worth discussing in an interview.

## Architecture

```
User query
   │
   ▼
LangChain agent (create_agent + tool loop)
   │
   ├── search_codebase ──► CodebaseRetriever ──► FAISS index
   ├── read_file
   ├── run_tests     ──► subprocess: pytest
   ├── run_linter    ──► subprocess: ruff
   └── git_diff      ──► subprocess: git diff
```

- **Chunking** (`src/ingest.py`): walks the repo, parses each `.py` file's AST, yields one chunk per function/class/module docstring — chosen over naive fixed-size or `RecursiveCharacterTextSplitter` chunking because it never splits a function in half.
- **Retrieval** (`src/retriever.py`): embeds each chunk as a `# file::name (type)` header + docstring + code (a docstring+code concatenation trick that measurably improved retrieval quality), indexes in FAISS.
- **Agent** (`src/agent.py`): `init_chat_model` + `create_agent` with the five tools above and a system prompt that forbids the agent from claiming it applied a fix it didn't actually apply.
- **Bug-risk classifier** (`train_bug_classifier.py`, `generate_bug_dataset_v3.py`): LoRA adapter (PEFT) on a frozen base transformer (DistilBERT → UniXcoder), trained on a synthetic dataset of AST-mutated bugs.

## Approach: how the retrieval side was chosen, not assumed

Rather than picking an embedding model or chunking strategy by reputation, each choice was backed by a small evaluation script, run and compared:

- **Chunking:** `compare_chunking.py` shows naive fixed-size chunking splitting functions mid-body; AST chunking never does.
- **Embedding model selection** (`eval_embeddings.py`): tested the intuitive pick — `microsoft/codebert-base`, a model specifically pretrained on code — against a general-purpose sentence embedder, `sentence-transformers/all-MiniLM-L6-v2`, on 10 hand-labeled (query, expected-function) pairs. Result: MiniLM **100%** top-3 accuracy vs. CodeBERT **0%**. CodeBERT wasn't trained with a contrastive/sentence-similarity objective, so its embeddings collapse (anisotropy) and are unusable for semantic search out of the box, despite being "for code." This is the kind of result that only shows up by actually testing an assumption.
- **Harder evaluation** (`eval_embeddings_hard.py`): a second, harder test set with near-duplicate confusable function pairs (`reverse_string` vs `reverse_list`, `divide` vs `safe_divide`) and indirect query phrasing that avoids reusing the function's own vocabulary, scored with top-1 accuracy + Mean Reciprocal Rank (MRR) instead of top-k pass/fail. Result: MiniLM and a code-specific alternative (`st-codesearch-distilroberta-base`) tied at 76.9% top-1, MiniLM with a slightly better MRR (0.885 vs 0.865) — kept MiniLM.

## Fine-tuning: the bug-risk classifier

### The idea

Generate synthetic "buggy" code by taking known-clean utility functions and mutating their AST (flip a comparison, remove a guard, swap arithmetic operators, swap boolean operators, silently swallow an error instead of raising, nudge a constant off by one, remove a scaling factor), then train a LoRA classifier to tell buggy from clean.

### What went wrong, in order, and how each was diagnosed and fixed

| # | Problem | How it was caught | Fix |
|---|---|---|---|
| 1 | **Data leakage** — near-duplicate mutants of the same base function could land in both train and test | No grouping key on the split | Added a `source_id` per base function; switched to `GroupShuffleSplit`/`GroupKFold`; asserted zero group overlap |
| 2 | **Underfitting** — model always predicted the majority class | Eval loss flat at ln(2) ≈ 0.693 for 8 straight epochs; recall exactly 1.0 | Raised learning rate 2e-4 → 1e-3 |
| 3 | **Overfitting** on a longer run | Eval loss reversed direction after ~epoch 8 | `save_strategy="epoch"` + `load_best_model_at_end=True` |
| 4 | **Metric gaming** — a trivial "always buggy" predictor scored the *highest* F1/accuracy of the whole run once the dataset was imbalanced | F1/accuracy both non-robust to class imbalance | Switched checkpoint-selection metric to balanced accuracy, which mathematically forces a constant predictor to exactly 0.5 |
| 5 | **Non-reproducibility** across "identical" reruns | Different results each run despite fixed code | `set_seed(42)` fixed most of it; the remainder was traced to Apple Silicon MPS non-determinism, fixed with `use_cpu=True` — verified byte-identical metrics across reruns afterward |
| 6 | **Weak generalization** to unfamiliar code | First out-of-distribution (OOD) test: 4/8 (50%), model just predicted "buggy" every time | Rebuilt the dataset larger and more diverse (v1 → v2) |
| 7 | A new mutation type (`delete_statement`) made things worse | Balanced accuracy dropped 0.664 → 0.572, OOD dropped 5/8 → 4/8 | Reverted, kept as a documented rejected experiment rather than silently dropped |
| 8 | Some bug categories had almost no training examples no matter how much data was generated | Traced to a structural bug: every mutator only ever mutated the *first* matching AST node per function | Rewrote the mutator to target any occurrence by index, not just the first (v2 → v3) |
| 9 | **Directional bias** — on the largest, most balanced OOD test, the model missed 8 of 12 real bugs while correctly clearing 10 of 12 clean functions | 24-example, exactly 12/12 balanced OOD test | Root-caused to inverse-frequency class weighting fighting the actual cost asymmetry of an audit tool (a missed bug costs more than a false alarm) — diagnosed, **not yet fixed** |

### Three rounds of out-of-distribution testing

Held-out eval and cross-validation only ever tested code from the same function banks the model trained on. To check whether it learned real patterns rather than memorizing those specific functions, I hand-wrote fresh test sets in domains it had never seen:

- **v1** (8 examples, 4 new domains): 4/8 (50%) — collapsed to predicting "buggy" every time.
- **v2** (8 examples, 3 new domains): 5/8 (62.5%) — real improvement, but also showed two checkpoints with similar in-distribution scores could differ wildly on OOD accuracy (62.5% vs. 25%).
- **v3** (24 examples, exactly 12/12 balanced, 12 new domains): 14/24 (58.3%) overall, but only **33% recall** on real bugs vs. 83% specificity on clean code — the overall number hid a serious under-flagging bias invisible in the smaller, unbalanced earlier tests.

### Per-mutation-type breakdown (5-fold cross-validation)

| Bug type | Category | Accuracy |
|---|---|---|
| Error silently swallowed instead of raised | Structural | 100% |
| Missing guard/bounds check | Structural | 74% |
| Off-by-one | Structural/numeric | 66% |
| Scaling factor removed | Semantic | 55% |
| Arithmetic operator swapped | Semantic | 54% |
| Comparison operator flipped | Semantic | 41% (below chance) |
| Boolean operator swapped | Semantic | 0% in v2 (n=3, too few); became measurable after the v3 mutator fix |

**Takeaway:** the model reliably catches bugs that change a function's *structure* — a missing guard, a swallowed error, an off-by-one index. It struggles with bugs that preserve structure but flip *meaning* — `a < b` vs `a > b` still looks like a normal, well-formed function; telling them apart requires understanding what the function is supposed to do, not just pattern-matching its shape. This is a plausible and defensible limitation of a small model trained on a few hundred examples, not a mysterious failure.

## Honest limitations

- The bug classifier is a demonstration of a sound *methodology*, not a production-ready detector. Its best OOD accuracy so far is 58.3%, with a known directional bias toward under-flagging bugs — the wrong direction for an audit gate.
- Root cause of that bias is understood (inverse-frequency class weighting) but not yet fixed. Two candidate fixes are identified and untested: lowering the inference-time decision threshold, or replacing inverse-frequency weights with weights that deliberately favor recall on buggy code.
- All training/eval data is synthetic (AST-mutated), not real-world bugs — the model's ceiling on actual production code is untested.
- Base models are small (66M–125M params) trained on small datasets (150–400 examples); the semantic-bug weakness may partly be a capacity/data ceiling rather than a fundamental one.
- The classifier is not yet wired into the agent as a callable tool, and the full audit-report generator (`run_tests` + `run_linter` + `check_test_coverage` + classifier → one structured report) is designed but not built.

## Tech stack

Python, LangChain / LangGraph (`create_agent`), Hugging Face `transformers` + `sentence-transformers`, FAISS, PEFT/LoRA, pytest, ruff, Gradio.

## Project structure

```
src/
  ingest.py                  # AST-based code chunker
  retriever.py                # FAISS retriever (HF embeddings)
  tools.py                     # Agent tools: search, read, test, lint, diff
  agent.py                      # Agent construction + system prompt
app.py                          # Gradio UI
sample_repo/                    # Demo repo with seeded bugs, used for evaluation
compare_chunking.py              # Naive vs. AST chunking demo
eval_embeddings.py               # Embedding model comparison (easy test set)
eval_embeddings_hard.py           # Embedding model comparison (hard test set, MRR)
generate_bug_dataset_v3.py         # Synthetic buggy-code dataset generator
train_bug_classifier.py             # LoRA fine-tuning script
cv_bug_classifier.py                 # 5-fold cross-validation + per-category breakdown
test_bug_classifier_v3.py             # 24-example out-of-distribution test
```

The Databricks integration (Delta Lake ingest notebook, Vector Search index, automated sync script) lives on the separate `databricks-integration` branch, not here.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY or ANTHROPIC_API_KEY, and MODEL_PROVIDER
python app.py
```

To reproduce the fine-tuning results:

```bash
python generate_bug_dataset_v3.py     # generates bug_dataset_v3.jsonl
python train_bug_classifier.py        # trains the LoRA classifier
python cv_bug_classifier.py           # 5-fold cross-validation + per-category breakdown
python test_bug_classifier_v3.py      # out-of-distribution test
```