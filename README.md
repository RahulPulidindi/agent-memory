# take-home

CLI chat agent built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents) with four cross-conversation memory strategies.

- **Results:** See [`harness_output.txt`](./harness_output.txt) for the latest evaluation run.
- **Write-up:** See [`WRITEUP.md`](./WRITEUP.md) for approach, trade-offs, and recommendation.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- `ANTHROPIC_API_KEY` — required for the default model and summary memory
- `OPENAI_API_KEY` — required for semantic memory embeddings

## Setup

```bash
git clone https://github.com/valkai-tech/take-home.git
cd take-home
uv sync
cp .env.example .env
# Add ANTHROPIC_API_KEY and OPENAI_API_KEY to .env
```

## Usage

```bash
# Stateless (no memory)
uv run chat

# With memory — pick a strategy and a user ID
uv run chat --memory history --user alice
uv run chat --memory summary --user alice
uv run chat --memory semantic --user alice
uv run chat --memory agentic --user alice

# Custom model or system prompt
uv run chat --model openai:gpt-4o --memory semantic --user alice
uv run chat --system "You are a concise infrastructure advisor." --memory history --user alice
```

Memory persists in `memory.db` between sessions. The same `--user` ID will load prior context on the next run. Type `quit` or `exit` to end a session.

## Evaluation harness

Runs a scripted three-session scenario against all five strategies and scores each one on recall accuracy, token efficiency, and latency overhead.

```bash
uv run python -u -m evals.harness | tee harness_output.txt
```

The harness creates a temporary `harness_memory.db` and clears it before each run.

## Tests

```bash
uv run pytest evals/ -v
```

Tests make real LLM API calls. Both API keys are required.
