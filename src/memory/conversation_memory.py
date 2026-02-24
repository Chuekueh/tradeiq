import sqlite3
import time
from pathlib import Path


class ConversationMemory:
    """SQLite-backed conversation memory for session persistence."""

    def __init__(self, db_path: str = "./data/conversations.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _init_db(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id)")
        self._conn.commit()

    def add_message(self, session_id: str, role: str, content: str) -> None:
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time()),
        )
        self._conn.commit()

    def get_history(self, session_id: str, limit: int = 20) -> list[dict[str, str]]:
        cursor = self._conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cursor.fetchall()
        return [{"role": role, "content": content} for role, content in reversed(rows)]

    def get_sessions(self) -> list[dict[str, str | int]]:
        cursor = self._conn.execute("""
            SELECT session_id, COUNT(*) as message_count, MAX(timestamp) as last_active
            FROM messages GROUP BY session_id ORDER BY last_active DESC
        """)
        return [
            {"session_id": row[0], "message_count": row[1], "last_active": row[2]}
            for row in cursor.fetchall()
        ]

    def clear_session(self, session_id: str) -> None:
        self._conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
