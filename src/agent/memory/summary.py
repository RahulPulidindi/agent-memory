import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from langchain.chat_models import init_chat_model


_SUMMARIZE_PROMPT = """\
You are a memory manager for an AI assistant. Maintain a concise, accurate running \
summary of what is known about a user from their conversations.

Current summary (empty if first conversation):
{current_summary}

New conversation turns to incorporate:
{new_turns}

Update the summary with important new information. Preserve specific technical details: \
role, team, project names, stack, tools, versions, constraints, and stated preferences. \
Drop pleasantries and filler. Return only the updated summary text, no preamble."""


class SummaryMemory:
    """Maintains a running LLM-generated summary of prior conversations.

    After each turn the summary is updated by merging new information into the
    existing one. Stays compact regardless of session count, but is lossy —
    fine-grained details may be dropped over time.
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        model_str: str = "anthropic:claude-haiku-4-5-20251001",
    ):
        self.db_path = Path(db_path)
        self.model_str = model_str
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    user_id    TEXT PRIMARY KEY,
                    summary    TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

    def retrieve(self, user_id: str, query: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM memory_summaries WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return f"Summary of prior conversations with this user:\n{row['summary']}" if row else ""

    def store(self, user_id: str, messages: list) -> None:
        new_turns = self._format_turns(messages)
        if not new_turns:
            return
        current = self._get_current_summary(user_id)
        updated = self._summarize(current, new_turns)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_summaries (user_id, summary, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET summary = excluded.summary, updated_at = excluded.updated_at
                """,
                (user_id, updated, now),
            )

    def clear(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM memory_summaries WHERE user_id = ?", (user_id,))

    def _get_current_summary(self, user_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary FROM memory_summaries WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["summary"] if row else ""

    def _format_turns(self, messages: list) -> str:
        lines = []
        for msg in messages:
            role = msg["role"] if isinstance(msg, dict) else msg.type
            content = msg["content"] if isinstance(msg, dict) else msg.content
            if role in ("user", "assistant") and content:
                lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
        return "\n".join(lines)

    def _summarize(self, current_summary: str, new_turns: str) -> str:
        model = init_chat_model(self.model_str)
        prompt = _SUMMARIZE_PROMPT.format(
            current_summary=current_summary or "(none yet)",
            new_turns=new_turns,
        )
        return model.invoke([{"role": "user", "content": prompt}]).content.strip()
