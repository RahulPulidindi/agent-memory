# take-home

Barebones CLI chat agent built on [LangChain Deep Agents](https://github.com/langchain-ai/deepagents). Supports OpenAI, Anthropic, and Google models out of the box. Designed as a minimal starting point to build on.

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- At least one LLM provider API key

## Quick start

```bash
git clone https://github.com/valkai-tech/take-home.git
cd take-home
uv sync
cp .env.example .env
# Fill in your API key(s) in .env
```

## Usage

```bash
# Default model (Anthropic Claude Haiku — cheapest)
uv run chat

# OpenAI
uv run chat --model openai:gpt-4o

# Google
uv run chat --model google_genai:gemini-2.5-flash

# Custom system prompt
uv run chat --system "You are a helpful coding assistant."
```

Type `quit` or `exit` to end the session.

## Running evals

```bash
uv run pytest evals/ -v
```

Evals make real LLM calls (not mocked) to verify provider integration end-to-end.

## Supported providers

| Provider  | Model string example                          | Required env var       |
|-----------|-----------------------------------------------|------------------------|
| Anthropic | `anthropic:claude-haiku-4-5-20251001` (default) | `ANTHROPIC_API_KEY`    |
| OpenAI    | `openai:gpt-4o`                               | `OPENAI_API_KEY`       |
| Google    | `google_genai:gemini-2.5-flash`               | `GOOGLE_API_KEY`       |

Any model supported by LangChain's [`init_chat_model`](https://docs.langchain.com/oss/python/langchain/models) works — just pass the `provider:model` string.

## Project structure

```
take-home/
├── pyproject.toml          # uv project config, dependencies
├── .env.example            # API key template
├── src/
│   └── agent/
│       ├── core.py         # Agent factory (create_deep_agent wrapper)
│       └── cli.py          # Interactive chat REPL (entry point)
└── evals/
    └── test_agent.py       # Barebones pytest evals
```
