"""Small SQLite store for conversation history and plant scan history."""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import CHAT_DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ChatStore:
    def __init__(self, path: Path | str = CHAT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def _init_db(self):
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL DEFAULT 'New chat',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_owner ON chat_sessions(owner_id, updated_at);
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL,
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON chat_messages(session_id, message_id);
                CREATE TABLE IF NOT EXISTS plant_scans (
                    scan_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    crop TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_healthy INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_scans_owner ON plant_scans(owner_id, scan_id DESC);
                """
            )

    def create_session(self, owner_id: str, title: str = "New chat") -> str:
        import uuid
        sid = uuid.uuid4().hex
        now = _now()
        with self._connect() as con:
            con.execute(
                "INSERT INTO chat_sessions(session_id, owner_id, title, created_at, updated_at) VALUES(?,?,?,?,?)",
                (sid, owner_id, title, now, now),
            )
        return sid

    def list_sessions(self, owner_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT session_id, title, created_at, updated_at FROM chat_sessions WHERE owner_id=? ORDER BY updated_at DESC",
                (owner_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_messages(self, session_id: str, owner_id: str) -> list[dict[str, Any]]:
        with self._connect() as con:
            allowed = con.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id=? AND owner_id=?",
                (session_id, owner_id),
            ).fetchone()
            if not allowed:
                return []
            rows = con.execute(
                "SELECT role, content, sources_json, created_at FROM chat_messages WHERE session_id=? ORDER BY message_id",
                (session_id,),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            try:
                item["sources"] = json.loads(item.pop("sources_json") or "[]")
            except json.JSONDecodeError:
                item["sources"] = []
            result.append(item)
        return result

    def add_message(self, session_id: str, owner_id: str, role: str, content: str, sources: list[dict] | None = None):
        now = _now()
        with self._connect() as con:
            allowed = con.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id=? AND owner_id=?",
                (session_id, owner_id),
            ).fetchone()
            if not allowed:
                raise ValueError("The conversation does not belong to this browser session.")
            con.execute(
                "INSERT INTO chat_messages(session_id, role, content, sources_json, created_at) VALUES(?,?,?,?,?)",
                (session_id, role, content, json.dumps(sources or [], ensure_ascii=False), now),
            )
            con.execute(
                "UPDATE chat_sessions SET updated_at=? WHERE session_id=?",
                (now, session_id),
            )
            row = con.execute("SELECT title FROM chat_sessions WHERE session_id=?", (session_id,)).fetchone()
            if role == "user" and row and row["title"] == "New chat":
                title = content.strip().replace("\n", " ")[:64] or "New chat"
                con.execute("UPDATE chat_sessions SET title=? WHERE session_id=?", (title, session_id))

    def delete_session(self, session_id: str, owner_id: str):
        with self._connect() as con:
            con.execute("DELETE FROM chat_sessions WHERE session_id=? AND owner_id=?", (session_id, owner_id))

    def save_scan(self, owner_id: str, filename: str, prediction: dict[str, Any]):
        with self._connect() as con:
            con.execute(
                "INSERT INTO plant_scans(owner_id, filename, predicted_label, crop, condition, confidence, is_healthy, created_at) VALUES(?,?,?,?,?,?,?,?)",
                (
                    owner_id, filename,
                    str(prediction.get("display_name") or prediction.get("class_name", "Unknown")),
                    str(prediction.get("crop", "Unknown")),
                    str(prediction.get("condition", "Unknown")),
                    float(prediction.get("confidence", 0.0)),
                    int(bool(prediction.get("is_healthy", False))),
                    _now(),
                ),
            )

    def list_scans(self, owner_id: str, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT filename, predicted_label, crop, condition, confidence, is_healthy, created_at FROM plant_scans WHERE owner_id=? ORDER BY scan_id DESC LIMIT ?",
                (owner_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]
