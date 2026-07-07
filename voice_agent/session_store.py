"""Per-call session state and transcript logging, backed by SQLite.

Keyed by Twilio's CallSid. Survives across the multiple HTTP requests Twilio
makes to /voice/gather during one call (each <Gather> round trip is a fresh
request), and doubles as the full call transcript log required for auditing
outcomes (booked/rescheduled/cancelled/handoff/abandoned).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class SessionState:
    call_sid: str
    caller_number: str
    stage: str = "start"  # start -> collecting -> confirming -> done
    action: Optional[str] = None  # book | reschedule | cancel
    slots: dict = field(default_factory=dict)
    target_appointment_id: Optional[int] = None
    turn_count: int = 0
    retry_count: int = 0
    outcome: Optional[str] = None  # booked | rescheduled | cancelled | handoff | abandoned

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "SessionState":
        return cls(**json.loads(raw))


class SessionStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                call_sid TEXT PRIMARY KEY,
                state_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transcript (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                call_sid TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get_or_create(self, call_sid: str, caller_number: str) -> SessionState:
        row = self._conn.execute(
            "SELECT state_json FROM sessions WHERE call_sid = ?", (call_sid,)
        ).fetchone()
        if row is not None:
            return SessionState.from_json(row["state_json"])
        state = SessionState(call_sid=call_sid, caller_number=caller_number)
        self.save(state)
        return state

    def save(self, state: SessionState) -> None:
        now = datetime.now().isoformat()
        self._conn.execute(
            """
            INSERT INTO sessions (call_sid, state_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(call_sid) DO UPDATE SET state_json = excluded.state_json, updated_at = excluded.updated_at
            """,
            (state.call_sid, state.to_json(), now, now),
        )
        self._conn.commit()

    def log_turn(self, call_sid: str, turn_index: int, speaker: str, text: str) -> None:
        self._conn.execute(
            "INSERT INTO transcript (call_sid, turn_index, speaker, text, created_at) VALUES (?, ?, ?, ?, ?)",
            (call_sid, turn_index, speaker, text, datetime.now().isoformat()),
        )
        self._conn.commit()

    def get_transcript(self, call_sid: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT turn_index, speaker, text, created_at FROM transcript WHERE call_sid = ? ORDER BY id",
            (call_sid,),
        ).fetchall()
        return [dict(r) for r in rows]

    def mark_abandoned_if_unresolved(self, call_sid: str) -> bool:
        """Called from the Twilio status callback when a call completes.

        Returns True if the session existed and had no outcome yet, in which
        case it is now marked 'abandoned'.
        """
        row = self._conn.execute(
            "SELECT state_json FROM sessions WHERE call_sid = ?", (call_sid,)
        ).fetchone()
        if row is None:
            return False
        state = SessionState.from_json(row["state_json"])
        if state.outcome is not None:
            return False
        state.outcome = "abandoned"
        self.save(state)
        return True

    def close(self) -> None:
        self._conn.close()
