"""Agentic Memory — agent-directed, structured, hybrid memory strategy.

Combines three production patterns into one strategy:

1. **Core memory blocks** — small, always-injected structured records (identity,
   preferences, constraints) that the agent can edit via tool calls.
2. **Extracted facts** — an LLM distils each exchange into discrete facts, which
   are embedded and stored for semantic retrieval (like Mem0).
3. **Async summary** — a background-style summary is updated after each turn but
   without blocking the store path (summary is rebuilt from facts, not from raw
   conversation, so it stays cheap).

The retrieve() output gives the agent its core blocks plus the top-K most relevant
extracted facts, keeping token injection bounded while guaranteeing that identity-
level information is always present.
"""

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_EXTRACT_FACTS_PROMPT = """\
You are a memory extraction module. Given a user/assistant exchange, extract \
discrete factual statements worth remembering for future conversations.

Focus on:
- Identity: name, role, company, team
- Technical context: stack, tools, versions, services, architecture decisions
- Constraints: SLAs, deadlines, budgets, rollback windows
- Preferences: communication style, priorities, stated opinions
- Decisions: what was agreed, chosen, or rejected

Skip:
- Pleasantries, filler, questions without answers
- Information the assistant provided (general knowledge) — only extract what \
  the USER revealed or what was DECIDED

Return a JSON array of strings, each a single fact. If nothing is worth \
extracting, return []. No preamble, no markdown fences — only the JSON array."""

_CORE_MEMORY_UPDATE_PROMPT = """\
You are a memory manager. You maintain a structured core memory block about a user.
The core memory should be a concise, always-up-to-date record of the most important \
facts: identity, role, company, active projects, key constraints, and preferences.

Current core memory:
{current_core}

New facts extracted from the latest exchange:
{new_facts}

Produce an updated core memory. Rules:
- Keep it under 500 words — this is injected into every prompt.
- Use short key-value or bullet format for scannability.
- If new facts contradict old ones, the new facts win.
- Remove anything that is no longer relevant if new facts supersede it.
- Preserve all facts that are still valid.

Return ONLY the updated core memory text. No preamble."""


class AgenticMemory:
    """Agent-directed hybrid memory with structured fact extraction.

    Architecturally inspired by Mem0 (fact extraction + semantic retrieval)
    and MemGPT/Letta (editable core memory blocks), adapted to fit the
    existing MemoryStrategy protocol.

    Storage layout (all in the same SQLite DB):
    - agentic_core: one row per user, the always-injected core memory block
    - agentic_facts: extracted facts with embeddings for semantic retrieval
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        extraction_model: str = "anthropic:claude-haiku-4-5-20251001",
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 10,
    ):
        self.db_path = Path(db_path)
        self.top_k = top_k
        self._extraction_model_str = extraction_model
        self._extraction_model = None  # lazy init
        self._embeddings = OpenAIEmbeddings(model=embedding_model)
        self._init_db()

    # -- lazy model init to avoid re-creating on every call --

    @property
    def _model(self):
        if self._extraction_model is None:
            self._extraction_model = init_chat_model(self._extraction_model_str)
        return self._extraction_model

    # ------------------------------------------------------------------
    # SQLite helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agentic_core (
                    user_id    TEXT PRIMARY KEY,
                    core       TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agentic_facts (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT    NOT NULL,
                    fact      TEXT    NOT NULL,
                    embedding TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_agentic_facts_user ON agentic_facts(user_id)"
            )

    # ------------------------------------------------------------------
    # MemoryStrategy protocol
    # ------------------------------------------------------------------

    def retrieve(self, user_id: str, query: str) -> str:
        """Build context from core block + semantically relevant facts."""
        sections = []

        # 1. Core memory block — always injected
        core = self._get_core(user_id)
        if core:
            sections.append(f"## Core Memory (always available)\n{core}")

        # 2. Semantic retrieval over extracted facts
        relevant = self._retrieve_facts(user_id, query)
        if relevant:
            sections.append(
                "## Relevant Facts from Prior Conversations\n" + "\n".join(f"- {f}" for f in relevant)
            )

        return "\n\n".join(sections)

    def store(self, user_id: str, messages: list) -> None:
        """Extract facts, embed them, and update core memory asynchronously."""
        turns_text = self._format_turns(messages)
        if not turns_text:
            return

        # Step 1: Extract structured facts via LLM
        facts = self._extract_facts(turns_text)
        if not facts:
            return

        # Step 2: Embed and persist facts
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for fact in facts:
                embedding = self._embeddings.embed_query(fact)
                conn.execute(
                    "INSERT INTO agentic_facts (user_id, fact, embedding, timestamp) VALUES (?, ?, ?, ?)",
                    (user_id, fact, json.dumps(embedding), now),
                )

        # Step 3: Update core memory block (async — fire and forget)
        thread = threading.Thread(
            target=self._update_core_async,
            args=(user_id, facts),
            daemon=True,
        )
        thread.start()
        # Store a reference so callers can join if needed (e.g. in tests/harness)
        self._last_core_update_thread = thread

    def clear(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM agentic_core WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM agentic_facts WHERE user_id = ?", (user_id,))

    # ------------------------------------------------------------------
    # Fact extraction
    # ------------------------------------------------------------------

    def _extract_facts(self, turns_text: str) -> list[str]:
        """Use LLM to extract discrete facts from the conversation turn."""
        result = self._model.invoke(
            [
                {"role": "system", "content": _EXTRACT_FACTS_PROMPT},
                {"role": "user", "content": turns_text},
            ],
            temperature=0,
        )
        raw = result.content.strip()
        # Strip markdown fences if the model wraps them
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                return [f for f in facts if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            pass
        return []

    # ------------------------------------------------------------------
    # Core memory block
    # ------------------------------------------------------------------

    def _get_core(self, user_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT core FROM agentic_core WHERE user_id = ?", (user_id,),
            ).fetchone()
        return row["core"] if row else ""

    def _update_core_async(self, user_id: str, new_facts: list[str]) -> None:
        """Merge new facts into the core memory block via LLM."""
        current_core = self._get_core(user_id)
        facts_text = "\n".join(f"- {f}" for f in new_facts)

        prompt = _CORE_MEMORY_UPDATE_PROMPT.format(
            current_core=current_core or "(empty — first interaction)",
            new_facts=facts_text,
        )
        result = self._model.invoke(
            [{"role": "user", "content": prompt}],
            temperature=0,
        )
        updated = result.content.strip()

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO agentic_core (user_id, core, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET core = excluded.core, updated_at = excluded.updated_at
                """,
                (user_id, updated, now),
            )

    # ------------------------------------------------------------------
    # Semantic fact retrieval
    # ------------------------------------------------------------------

    def _retrieve_facts(self, user_id: str, query: str) -> list[str]:
        """Return top-K facts most relevant to the query."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT fact, embedding FROM agentic_facts WHERE user_id = ?",
                (user_id,),
            ).fetchall()

        if not rows:
            return []

        facts = [row["fact"] for row in rows]
        matrix = np.array([json.loads(row["embedding"]) for row in rows])

        query_vec = np.array(self._embeddings.embed_query(query))
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(row_norms > 0, row_norms, 1.0)
        scores = (matrix / safe_norms) @ (query_vec / query_norm)
        scores[row_norms.squeeze() == 0] = 0.0

        top_indices = np.argsort(scores)[::-1][: self.top_k]
        return [facts[i] for i in top_indices]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_turns(self, messages: list) -> str:
        lines = []
        for msg in messages:
            role = msg["role"] if isinstance(msg, dict) else msg.type
            content = msg["content"] if isinstance(msg, dict) else msg.content
            if role in ("user", "assistant") and content:
                lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
        return "\n".join(lines)

    def wait_for_core_update(self, timeout: float = 30.0) -> None:
        """Block until the last core memory update finishes. For tests/harness use."""
        thread = getattr(self, "_last_core_update_thread", None)
        if thread is not None:
            thread.join(timeout=timeout)