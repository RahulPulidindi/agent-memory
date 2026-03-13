import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class FullHistoryMemory:
    """Stores every message verbatim and replays the full transcript on retrieval.

    Lossless, but grows without bound — will eventually exceed context limits
    for long-lived users.
    """

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS history_messages (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT    NOT NULL,
                    role      TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON history_messages(user_id)")

    def retrieve(self, user_id: str, query: str) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM history_messages WHERE user_id = ? ORDER BY id",
                (user_id,),
            ).fetchall()

        if not rows:
            return ""

        lines = [
            f"{'User' if row['role'] == 'user' else 'Assistant'}: {row['content']}"
            for row in rows
        ]
        return "Full conversation history from prior sessions:\n" + "\n".join(lines)

    def store(self, user_id: str, messages: list) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for msg in messages:
                role = msg["role"] if isinstance(msg, dict) else msg.type
                content = msg["content"] if isinstance(msg, dict) else msg.content
                if role in ("user", "assistant") and content:
                    conn.execute(
                        "INSERT INTO history_messages (user_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                        (user_id, role, content, now),
                    )

    def clear(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM history_messages WHERE user_id = ?", (user_id,))
