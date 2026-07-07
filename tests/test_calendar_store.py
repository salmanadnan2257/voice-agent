"""Pure calendar-store tests: no network, no Twilio, no Gemini. Every test
gets its own temporary SQLite file so tests never share state."""
from __future__ import annotations

from datetime import datetime

import pytest

from voice_agent.calendar_store import ConflictError, NotFoundError, SqliteCalendarStore


@pytest.fixture
def store(tmp_path):
    s = SqliteCalendarStore(str(tmp_path / "calendar.db"))
    yield s
    s.close()


def test_book_creates_appointment(store):
    appt = store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    assert appt.id is not None
    assert appt.name == "Alice"
    assert appt.status == "booked"
    fetched = store.get(appt.id)
    assert fetched.start_time == datetime(2026, 8, 1, 10, 0)


def test_book_rejects_exact_overlap(store):
    store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    with pytest.raises(ConflictError):
        store.book("Bob", "+15551110001", datetime(2026, 8, 1, 10, 0), 30)


def test_book_rejects_partial_overlap(store):
    store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    with pytest.raises(ConflictError):
        store.book("Bob", "+15551110001", datetime(2026, 8, 1, 10, 15), 30)


def test_book_allows_back_to_back_slots(store):
    store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    # Starts exactly when the first ends: not an overlap.
    second = store.book("Bob", "+15551110001", datetime(2026, 8, 1, 10, 30), 30)
    assert second.id is not None


def test_cancelled_appointment_frees_the_slot(store):
    appt = store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    store.cancel(appt.id)
    # Same slot should now be free.
    second = store.book("Bob", "+15551110001", datetime(2026, 8, 1, 10, 0), 30)
    assert second.id is not None


def test_reschedule_moves_appointment(store):
    appt = store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    moved = store.reschedule(appt.id, datetime(2026, 8, 2, 11, 0))
    assert moved.start_time == datetime(2026, 8, 2, 11, 0)
    assert moved.duration_minutes == 30


def test_reschedule_excludes_its_own_slot_from_conflict_check(store):
    appt = store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    # Rescheduling to the exact same slot must not conflict with itself.
    moved = store.reschedule(appt.id, datetime(2026, 8, 1, 10, 0))
    assert moved.id == appt.id


def test_reschedule_rejects_conflict_with_other_appointment(store):
    store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    bob = store.book("Bob", "+15551110001", datetime(2026, 8, 1, 11, 0), 30)
    with pytest.raises(ConflictError):
        store.reschedule(bob.id, datetime(2026, 8, 1, 10, 0))


def test_cancel_nonexistent_raises(store):
    with pytest.raises(NotFoundError):
        store.cancel(999)


def test_get_nonexistent_raises(store):
    with pytest.raises(NotFoundError):
        store.get(999)


def test_list_by_phone_active_only_excludes_cancelled(store):
    appt = store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    store.cancel(appt.id)
    assert store.list_by_phone("+15551110000", active_only=True) == []
    assert len(store.list_by_phone("+15551110000", active_only=False)) == 1


def test_list_by_phone_orders_most_recent_start_first(store):
    store.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    store.book("Alice", "+15551110000", datetime(2026, 8, 3, 10, 0), 30)
    results = store.list_by_phone("+15551110000")
    assert results[0].start_time == datetime(2026, 8, 3, 10, 0)


def test_schema_persists_across_reconnect(tmp_path):
    path = str(tmp_path / "calendar.db")
    store1 = SqliteCalendarStore(path)
    appt = store1.book("Alice", "+15551110000", datetime(2026, 8, 1, 10, 0), 30)
    store1.close()

    store2 = SqliteCalendarStore(path)
    fetched = store2.get(appt.id)
    assert fetched.name == "Alice"
    store2.close()
