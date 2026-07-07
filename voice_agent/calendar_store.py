"""Calendar storage: SQLite by default, an optional Google Calendar adapter.

Both implementations satisfy the same ``CalendarStore`` interface so the rest
of the app (conversation.py, app.py) never needs to know which backend is
active. ``get_calendar_store()`` picks the backend based on whether
``GOOGLE_CALENDAR_CREDENTIALS_PATH`` is set; the SQLite backend needs no
credentials and is what runs by default and in every test in this repo.
"""
from __future__ import annotations

import abc
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional


@dataclass
class Appointment:
    id: int
    name: str
    phone: str
    start_time: datetime
    duration_minutes: int
    status: str  # "booked" or "cancelled"

    @property
    def end_time(self) -> datetime:
        return self.start_time + timedelta(minutes=self.duration_minutes)


class ConflictError(RuntimeError):
    """Raised when a requested slot overlaps an existing booked appointment."""

    def __init__(self, conflicting: Appointment):
        self.conflicting = conflicting
        super().__init__(
            f"Slot overlaps existing appointment #{conflicting.id} "
            f"({conflicting.start_time.isoformat()} for {conflicting.duration_minutes}m)"
        )


class NotFoundError(RuntimeError):
    """Raised when an appointment lookup fails."""


class CalendarStore(abc.ABC):
    """Interface both the SQLite store and the Google Calendar adapter implement."""

    @abc.abstractmethod
    def find_conflict(
        self, start: datetime, duration_minutes: int, exclude_id: Optional[int] = None
    ) -> Optional[Appointment]:
        """Return the first booked appointment overlapping the given slot, if any."""

    @abc.abstractmethod
    def book(self, name: str, phone: str, start: datetime, duration_minutes: int) -> Appointment:
        """Create a new booked appointment. Raises ConflictError on overlap."""

    @abc.abstractmethod
    def get(self, appointment_id: int) -> Appointment:
        """Raises NotFoundError if the id doesn't exist."""

    @abc.abstractmethod
    def list_by_phone(self, phone: str, active_only: bool = True) -> list[Appointment]:
        """All appointments for a phone number, most recent first."""

    @abc.abstractmethod
    def reschedule(self, appointment_id: int, new_start: datetime, new_duration_minutes: Optional[int] = None) -> Appointment:
        """Move an existing appointment to a new slot. Raises ConflictError/NotFoundError."""

    @abc.abstractmethod
    def cancel(self, appointment_id: int) -> Appointment:
        """Mark an appointment cancelled. Raises NotFoundError."""


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    """Half-open interval overlap: touching endpoints (a_end == b_start) is NOT a conflict."""
    return a_start < b_end and b_start < a_end


class SqliteCalendarStore(CalendarStore):
    """Default calendar backend. One appointments table, real conflict detection."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS appointments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                start_time TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                status TEXT NOT NULL DEFAULT 'booked',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def _row_to_appointment(self, row: sqlite3.Row) -> Appointment:
        return Appointment(
            id=row["id"],
            name=row["name"],
            phone=row["phone"],
            start_time=datetime.fromisoformat(row["start_time"]),
            duration_minutes=row["duration_minutes"],
            status=row["status"],
        )

    def find_conflict(
        self, start: datetime, duration_minutes: int, exclude_id: Optional[int] = None
    ) -> Optional[Appointment]:
        end = start + timedelta(minutes=duration_minutes)
        cur = self._conn.execute(
            "SELECT * FROM appointments WHERE status = 'booked'"
            + (" AND id != ?" if exclude_id is not None else ""),
            (exclude_id,) if exclude_id is not None else (),
        )
        for row in cur.fetchall():
            appt = self._row_to_appointment(row)
            if _overlaps(start, end, appt.start_time, appt.end_time):
                return appt
        return None

    def book(self, name: str, phone: str, start: datetime, duration_minutes: int) -> Appointment:
        conflict = self.find_conflict(start, duration_minutes)
        if conflict is not None:
            raise ConflictError(conflict)
        now = datetime.now().isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO appointments (name, phone, start_time, duration_minutes, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, 'booked', ?, ?)
            """,
            (name, phone, start.isoformat(), duration_minutes, now, now),
        )
        self._conn.commit()
        return self.get(cur.lastrowid)

    def get(self, appointment_id: int) -> Appointment:
        row = self._conn.execute(
            "SELECT * FROM appointments WHERE id = ?", (appointment_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"No appointment with id {appointment_id}")
        return self._row_to_appointment(row)

    def list_by_phone(self, phone: str, active_only: bool = True) -> list[Appointment]:
        query = "SELECT * FROM appointments WHERE phone = ?"
        if active_only:
            query += " AND status = 'booked'"
        query += " ORDER BY start_time DESC"
        rows = self._conn.execute(query, (phone,)).fetchall()
        return [self._row_to_appointment(r) for r in rows]

    def reschedule(
        self, appointment_id: int, new_start: datetime, new_duration_minutes: Optional[int] = None
    ) -> Appointment:
        existing = self.get(appointment_id)
        if existing.status != "booked":
            raise NotFoundError(f"Appointment {appointment_id} is not active (status={existing.status})")
        duration = new_duration_minutes or existing.duration_minutes
        conflict = self.find_conflict(new_start, duration, exclude_id=appointment_id)
        if conflict is not None:
            raise ConflictError(conflict)
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE appointments SET start_time = ?, duration_minutes = ?, updated_at = ? WHERE id = ?",
            (new_start.isoformat(), duration, now, appointment_id),
        )
        self._conn.commit()
        return self.get(appointment_id)

    def cancel(self, appointment_id: int) -> Appointment:
        existing = self.get(appointment_id)
        now = datetime.now().isoformat()
        self._conn.execute(
            "UPDATE appointments SET status = 'cancelled', updated_at = ? WHERE id = ?",
            (now, appointment_id),
        )
        self._conn.commit()
        return self.get(appointment_id)

    def close(self) -> None:
        self._conn.close()


class GoogleCalendarStore(CalendarStore):
    """Optional adapter used only when GOOGLE_CALENDAR_CREDENTIALS_PATH is set.

    Talks to the real Google Calendar API v3 (events.insert/patch/delete,
    freebusy.query for conflict detection) against a single calendar id.
    Not exercised against a live calendar in this repo's tests: no Google
    Calendar credentials were provided in .env (only the SQLite path was
    verified end to end). Implemented fully so it is not a stub; wiring it up
    is a matter of setting the credentials path and CALENDAR_ID env var.
    """

    def __init__(self, credentials_path: str, calendar_id: Optional[str] = None):
        # Imported lazily: google-api-python-client is only needed if this
        # adapter is actually selected, so the SQLite-only default install
        # doesn't require it.
        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        self.calendar_id = calendar_id or os.environ.get("GOOGLE_CALENDAR_ID", "primary")
        creds = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=["https://www.googleapis.com/auth/calendar"],
        )
        self._service = build("calendar", "v3", credentials=creds, cacheDiscovery=False)

    def find_conflict(
        self, start: datetime, duration_minutes: int, exclude_id: Optional[int] = None
    ) -> Optional[Appointment]:
        end = start + timedelta(minutes=duration_minutes)
        body = {
            "timeMin": start.isoformat(),
            "timeMax": end.isoformat(),
            "items": [{"id": self.calendar_id}],
        }
        result = self._service.freebusy().query(body=body).execute()
        busy_periods = result["calendars"][self.calendar_id]["busy"]
        for period in busy_periods:
            event_start = datetime.fromisoformat(period["start"].replace("Z", "+00:00"))
            event_end = datetime.fromisoformat(period["end"].replace("Z", "+00:00"))
            if _overlaps(start, end, event_start, event_end):
                events = self._service.events().list(
                    calendarId=self.calendar_id,
                    timeMin=period["start"],
                    timeMax=period["end"],
                    singleEvents=True,
                ).execute()
                items = events.get("items", [])
                event_id = items[0]["id"] if items else "unknown"
                if exclude_id is not None and str(exclude_id) == event_id:
                    continue
                return Appointment(
                    id=event_id,
                    name=items[0].get("summary", "") if items else "",
                    phone="",
                    start_time=event_start,
                    duration_minutes=int((event_end - event_start).total_seconds() // 60),
                    status="booked",
                )
        return None

    def book(self, name: str, phone: str, start: datetime, duration_minutes: int) -> Appointment:
        conflict = self.find_conflict(start, duration_minutes)
        if conflict is not None:
            raise ConflictError(conflict)
        end = start + timedelta(minutes=duration_minutes)
        event = {
            "summary": f"Appointment: {name}",
            "description": f"Phone: {phone}",
            "start": {"dateTime": start.isoformat()},
            "end": {"dateTime": end.isoformat()},
        }
        created = self._service.events().insert(calendarId=self.calendar_id, body=event).execute()
        return Appointment(
            id=created["id"],
            name=name,
            phone=phone,
            start_time=start,
            duration_minutes=duration_minutes,
            status="booked",
        )

    def get(self, appointment_id) -> Appointment:
        event = self._service.events().get(calendarId=self.calendar_id, eventId=str(appointment_id)).execute()
        start = datetime.fromisoformat(event["start"]["dateTime"])
        end = datetime.fromisoformat(event["end"]["dateTime"])
        description = event.get("description", "")
        phone = description.replace("Phone:", "").strip() if "Phone:" in description else ""
        return Appointment(
            id=event["id"],
            name=event.get("summary", "").replace("Appointment: ", ""),
            phone=phone,
            start_time=start,
            duration_minutes=int((end - start).total_seconds() // 60),
            status="cancelled" if event.get("status") == "cancelled" else "booked",
        )

    def list_by_phone(self, phone: str, active_only: bool = True) -> list[Appointment]:
        events = self._service.events().list(
            calendarId=self.calendar_id, q=phone, singleEvents=True, orderBy="startTime"
        ).execute()
        results = []
        for item in events.get("items", []):
            appt = self.get(item["id"])
            if appt.phone == phone and (not active_only or appt.status == "booked"):
                results.append(appt)
        return results

    def reschedule(self, appointment_id, new_start: datetime, new_duration_minutes: Optional[int] = None) -> Appointment:
        existing = self.get(appointment_id)
        duration = new_duration_minutes or existing.duration_minutes
        conflict = self.find_conflict(new_start, duration, exclude_id=appointment_id)
        if conflict is not None:
            raise ConflictError(conflict)
        end = new_start + timedelta(minutes=duration)
        self._service.events().patch(
            calendarId=self.calendar_id,
            eventId=str(appointment_id),
            body={"start": {"dateTime": new_start.isoformat()}, "end": {"dateTime": end.isoformat()}},
        ).execute()
        return self.get(appointment_id)

    def cancel(self, appointment_id) -> Appointment:
        self._service.events().delete(calendarId=self.calendar_id, eventId=str(appointment_id)).execute()
        # Google Calendar delete removes the event outright; represent it to
        # the rest of the app as a cancelled appointment snapshot.
        return Appointment(
            id=appointment_id,
            name="",
            phone="",
            start_time=datetime.now(),
            duration_minutes=0,
            status="cancelled",
        )


def get_calendar_store(db_path: Optional[str] = None) -> CalendarStore:
    """Factory: Google Calendar adapter if configured, else the SQLite default."""
    from . import config

    creds_path = config.google_calendar_credentials_path()
    if creds_path and os.path.exists(creds_path):
        return GoogleCalendarStore(creds_path)
    return SqliteCalendarStore(db_path or config.db_path())
