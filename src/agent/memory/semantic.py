import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from langchain_openai import OpenAIEmbeddings


class SemanticMemory:
    """Retrieves the most relevant past exchanges via embedding cosine similarity.

    Each user/assistant exchange is embedded and stored. On retrieval, only the
    top-K most similar chunks are returned, keeping injection size bounded
    regardless of how much history accumulates.

    Uses SQLite for storage and numpy for batched similarity — no vector DB required.
    """

    def __init__(
        self,
        db_path: str = "memory.db",
        model: str = "text-embedding-3-small",
        top_k: int = 5,
    ):
        self.db_path = Path(db_path)
        self.top_k = top_k
        self._embeddings = OpenAIEmbeddings(model=model)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memories (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   TEXT    NOT NULL,
                    content   TEXT    NOT NULL,
                    embedding TEXT    NOT NULL,
                    timestamp TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_semantic_user ON semantic_memories(user_id)"
            )

    def retrieve(self, user_id: str, query: str) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT content, embedding FROM semantic_memories WHERE user_id = ?",
                (user_id,),
            ).fetchall()

        if not rows:
            return ""

        contents = [row["content"] for row in rows]
        matrix = np.array([json.loads(row["embedding"]) for row in rows])

        query_vec = np.array(self._embeddings.embed_query(query))
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return ""

        # Batched cosine similarity: normalise rows and query, then dot product.
        row_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        safe_norms = np.where(row_norms > 0, row_norms, 1.0)
        scores = (matrix / safe_norms) @ (query_vec / query_norm)
        scores[row_norms.squeeze() == 0] = 0.0

        top_indices = np.argsort(scores)[::-1][: self.top_k]
        top = [contents[i] for i in top_indices]
        return "Relevant context from prior conversations:\n" + "\n---\n".join(top)

    def store(self, user_id: str, messages: list) -> None:
        pairs = self._extract_pairs(messages)
        if not pairs:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for content in pairs:
                embedding = self._embeddings.embed_query(content)
                conn.execute(
                    "INSERT INTO semantic_memories (user_id, content, embedding, timestamp) VALUES (?, ?, ?, ?)",
                    (user_id, content, json.dumps(embedding), now),
                )

    def clear(self, user_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM semantic_memories WHERE user_id = ?", (user_id,))

    def _extract_pairs(self, messages: list) -> list[str]:
        """Combine consecutive user/assistant messages into single storable chunks."""
        pairs = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            role = msg["role"] if isinstance(msg, dict) else msg.type
            content = msg["content"] if isinstance(msg, dict) else msg.content

            if role == "user" and content:
                user_text = content
                assistant_text = ""
                if i + 1 < len(messages):
                    nxt = messages[i + 1]
                    nxt_role = nxt["role"] if isinstance(nxt, dict) else nxt.type
                    nxt_content = nxt["content"] if isinstance(nxt, dict) else nxt.content
                    if nxt_role == "assistant" and nxt_content:
                        assistant_text = nxt_content
                        i += 1

                chunk = f"User: {user_text}"
                if assistant_text:
                    chunk += f"\nAssistant: {assistant_text}"
                pairs.append(chunk)
            i += 1

        return pairs
