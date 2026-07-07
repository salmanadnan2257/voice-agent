"""FastAPI endpoint tests via TestClient. No real Twilio calls happen here:
Twilio is never invoked in either direction, these tests just exercise the
same HTTP endpoints Twilio would call, with the real conversation manager
and a real temporary SQLite calendar, but the free ScriptedIntentEngine
standing in for Gemini.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from eval.scenarios import FIXED_NOW
from eval.scripted_engine import ScriptedIntentEngine
from voice_agent.app import create_app
from voice_agent.calendar_store import SqliteCalendarStore
from voice_agent.conversation import ConversationManager
from voice_agent.session_store import SessionStore


@pytest.fixture
def client(tmp_path):
    calendar_store = SqliteCalendarStore(str(tmp_path / "calendar.db"))
    session_store = SessionStore(str(tmp_path / "session.db"))
    engine = ScriptedIntentEngine()
    app = create_app(
        calendar_store=calendar_store,
        session_store=session_store,
        intent_engine=engine,
        enforce_signature=False,
    )
    # Swap in a conversation manager with a fixed "now" so date/time parsing
    # and business-hours checks are deterministic in tests.
    app.state.conversation_manager = ConversationManager(
        intent_engine=engine,
        calendar_store=calendar_store,
        session_store=session_store,
        now_fn=lambda: FIXED_NOW,
    )
    with TestClient(app) as c:
        yield c, calendar_store, session_store


def test_health(client):
    c, _, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_voice_incoming_returns_gather_twiml(client):
    c, _, _ = client
    resp = c.post(
        "/voice/incoming",
        data={"CallSid": "CA_test_1", "From": "+15551234567", "To": "+15559999999"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")
    root = ET.fromstring(resp.text)
    assert root.tag == "Response"
    gather = root.find("Gather")
    assert gather is not None
    # No PUBLIC_WEBHOOK_BASE_URL is configured in this test, so the action
    # falls back to whatever host the request actually arrived on.
    assert gather.attrib["action"].endswith("/voice/gather")
    say = gather.find("Say")
    assert say is not None and "book" in say.text.lower()


def test_voice_gather_unclear_reprompts_without_ending_call(client):
    c, _, _ = client
    call_sid = "CA_test_unclear"
    c.post("/voice/incoming", data={"CallSid": call_sid, "From": "+15551234567", "To": "+15559999999"})
    resp = c.post("/voice/gather", data={"CallSid": call_sid, "From": "+15551234567", "SpeechResult": ""})
    assert resp.status_code == 200
    root = ET.fromstring(resp.text)
    # Still gathering, not hung up: no bare top-level <Hangup> right after <Say>.
    assert root.find("Gather") is not None


def test_full_booking_flow_via_gather_updates_calendar(client):
    c, calendar_store, session_store = client
    call_sid = "CA_test_booking"
    from_number = "+15557654321"

    c.post("/voice/incoming", data={"CallSid": call_sid, "From": from_number, "To": "+15559999999"})

    turns = [
        "I'd like to book an appointment. My name is Priya Rao, July 15th at 2pm.",
        "yes",
    ]
    last_resp = None
    for speech in turns:
        last_resp = c.post(
            "/voice/gather", data={"CallSid": call_sid, "From": from_number, "SpeechResult": speech}
        )
        assert last_resp.status_code == 200

    root = ET.fromstring(last_resp.text)
    assert root.find("Hangup") is None or root.find("Gather") is not None  # last turn keeps gathering ("anything else?")

    appts = calendar_store.list_by_phone(from_number)
    assert len(appts) == 1
    assert appts[0].name == "Priya Rao"
    assert appts[0].start_time == datetime(2026, 7, 15, 14, 0)

    transcript = session_store.get_transcript(call_sid)
    assert len(transcript) >= 4  # greeting + 2 caller turns + at least 2 agent replies


def test_voice_gather_ends_call_with_hangup_on_goodbye(client):
    c, calendar_store, session_store = client
    call_sid = "CA_test_goodbye"
    from_number = "+15550001111"

    c.post("/voice/incoming", data={"CallSid": call_sid, "From": from_number, "To": "+15559999999"})
    c.post("/voice/gather", data={"CallSid": call_sid, "From": from_number, "SpeechResult": "I want to cancel my appointment"})
    resp = c.post(
        "/voice/gather",
        data={"CallSid": call_sid, "From": from_number, "SpeechResult": "no, nothing else, bye"},
    )
    root = ET.fromstring(resp.text)
    assert root.find("Hangup") is not None
    assert root.find("Gather") is None


def test_voice_status_marks_abandoned_when_unresolved(client):
    c, _, session_store = client
    call_sid = "CA_test_status"
    from_number = "+15550002222"

    c.post("/voice/incoming", data={"CallSid": call_sid, "From": from_number, "To": "+15559999999"})
    c.post(
        "/voice/gather",
        data={"CallSid": call_sid, "From": from_number, "SpeechResult": "I'd like to book an appointment"},
    )
    resp = c.post("/voice/status", data={"CallSid": call_sid, "CallStatus": "completed"})
    assert resp.status_code == 204

    session = session_store.get_or_create(call_sid, from_number)
    assert session.outcome == "abandoned"


def test_voice_status_does_not_overwrite_a_real_outcome(client):
    c, calendar_store, session_store = client
    call_sid = "CA_test_status_booked"
    from_number = "+15550003333"

    c.post("/voice/incoming", data={"CallSid": call_sid, "From": from_number, "To": "+15559999999"})
    c.post(
        "/voice/gather",
        data={
            "CallSid": call_sid,
            "From": from_number,
            "SpeechResult": "I'd like to book an appointment. My name is Sam Kessler, July 15th at 3pm.",
        },
    )
    c.post("/voice/gather", data={"CallSid": call_sid, "From": from_number, "SpeechResult": "yes"})
    c.post("/voice/status", data={"CallSid": call_sid, "CallStatus": "completed"})

    session = session_store.get_or_create(call_sid, from_number)
    assert session.outcome == "booked"
