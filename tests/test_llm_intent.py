"""A small, deliberately kept-short set of tests against the REAL Gemini
2.5 Flash intent engine (voice_agent.intent_engine.GeminiIntentEngine), on
the live Vertex AI credential in .env. Each test function is exactly one
real, billed API call. Per the project's budget discipline this file is
capped at 6 calls total; see the README for the actual cost incurred on the
last real run.

Run only these with: pytest tests/test_llm_intent.py -m real_api
Skip these (e.g. for a fast/free local loop) with: pytest -m "not real_api"
"""
from __future__ import annotations

from datetime import date

import pytest
from dotenv import load_dotenv

from voice_agent import config
from voice_agent.intent_engine import GeminiIntentEngine, TurnContext

load_dotenv()

pytestmark = pytest.mark.real_api


@pytest.fixture(scope="module")
def engine():
    try:
        gemini_cfg = config.load_gemini_config()
    except config.ConfigError:
        pytest.skip("Gemini/Vertex credentials not configured")
    return GeminiIntentEngine(project=gemini_cfg.project, location=gemini_cfg.location)


def test_real_gemini_classifies_a_clear_booking_request(engine):
    ctx = TurnContext(
        caller_number="+15551234567",
        utterance="Hi, I'd like to book an appointment for this Friday at 2pm, my name is Jordan Lee.",
        history=[{"speaker": "agent", "text": "What would you like to do?"}],
        known_slots={},
        current_action=None,
        today=date(2026, 7, 6),
    )
    result = engine.parse_turn(ctx)
    assert result.intent == "book"
    assert result.name and "jordan" in result.name.lower()
    assert result.time_24h == "14:00"


def test_real_gemini_classifies_a_reschedule_request(engine):
    ctx = TurnContext(
        caller_number="+15551234567",
        utterance="I need to move my appointment to next Monday morning instead.",
        history=[{"speaker": "agent", "text": "What would you like to do?"}],
        known_slots={},
        current_action=None,
        today=date(2026, 7, 6),
    )
    result = engine.parse_turn(ctx)
    assert result.intent == "reschedule"


def test_real_gemini_classifies_a_cancel_request(engine):
    ctx = TurnContext(
        caller_number="+15551234567",
        utterance="Please cancel my appointment, something came up.",
        history=[{"speaker": "agent", "text": "What would you like to do?"}],
        known_slots={},
        current_action=None,
        today=date(2026, 7, 6),
    )
    result = engine.parse_turn(ctx)
    assert result.intent == "cancel"


def test_real_gemini_flags_gibberish_as_unclear(engine):
    ctx = TurnContext(
        caller_number="+15551234567",
        utterance="uhh mmphf static garble not really words",
        history=[{"speaker": "agent", "text": "What would you like to do?"}],
        known_slots={},
        current_action=None,
        today=date(2026, 7, 6),
    )
    result = engine.parse_turn(ctx)
    assert result.intent == "unclear"


def test_real_gemini_reads_back_confirmation_yes(engine):
    ctx = TurnContext(
        caller_number="+15551234567",
        utterance="Yes, that's correct, please go ahead and book it.",
        history=[
            {"speaker": "agent", "text": "So that's Jordan Lee on Friday July 10th at 2pm. Should I book it?"}
        ],
        known_slots={"name": "Jordan Lee", "date_iso": "2026-07-10", "time_24h": "14:00"},
        current_action="book",
        today=date(2026, 7, 6),
    )
    result = engine.parse_turn(ctx)
    assert result.intent == "book"
    assert result.confirmed is True


def test_real_gemini_reads_back_confirmation_no(engine):
    ctx = TurnContext(
        caller_number="+15551234567",
        utterance="No, that's wrong, I actually wanted 3pm not 2pm.",
        history=[
            {"speaker": "agent", "text": "So that's Jordan Lee on Friday July 10th at 2pm. Should I book it?"}
        ],
        known_slots={"name": "Jordan Lee", "date_iso": "2026-07-10", "time_24h": "14:00"},
        current_action="book",
        today=date(2026, 7, 6),
    )
    result = engine.parse_turn(ctx)
    assert result.intent == "book"
    assert result.confirmed is False
    assert result.time_24h == "15:00"
