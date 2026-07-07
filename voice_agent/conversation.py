"""The conversation state machine: wires the intent engine, the calendar
store and per-call session state together into the book/reschedule/cancel
business logic (slot filling, confirmation, conflict handling, business
hours, retries, handoff).

This module never talks to Twilio or Gemini directly; it depends only on the
``IntentEngine`` and ``CalendarStore`` interfaces, which is what makes it
possible to run the exact same logic in three places: the live FastAPI app
(real Gemini + real SQLite), the pytest TestClient tests (scripted engine),
and the offline evaluation harness in eval/scenarios.py (scripted engine).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

from . import config
from .calendar_store import CalendarStore, ConflictError, NotFoundError
from .intent_engine import IntentEngine, IntentEngineError, TurnContext
from .session_store import SessionState, SessionStore

logger = logging.getLogger("voice_agent.conversation")

SLOT_KEYS = ("name", "phone", "date_iso", "time_24h", "duration_minutes")

HANDOFF_MESSAGE = (
    "I'm having trouble understanding, so let me connect you with our office directly. "
    "Someone will be able to help you from here. Goodbye."
)
TECH_FAILURE_MESSAGE = (
    "We're experiencing a technical issue on our end. Please try your call again in a "
    "few minutes, or contact our office directly. Goodbye."
)
REPEAT_MESSAGE = "Sorry, I didn't quite catch that. Could you say that again?"


@dataclass
class TurnOutcome:
    reply: str
    end_call: bool
    outcome: Optional[str] = None  # set only when the call has just reached a terminal state


def _parse_slot_datetime(date_iso: Optional[str], time_24h: Optional[str]) -> Optional[datetime]:
    if not date_iso or not time_24h:
        return None
    try:
        return datetime.strptime(f"{date_iso} {time_24h}", "%Y-%m-%d %H:%M")
    except ValueError:
        return None


class ConversationManager:
    def __init__(
        self,
        intent_engine: IntentEngine,
        calendar_store: CalendarStore,
        session_store: SessionStore,
        now_fn: Callable[[], datetime] = datetime.now,
    ):
        self.intent_engine = intent_engine
        self.calendar_store = calendar_store
        self.session_store = session_store
        self.now_fn = now_fn

    def _clean_slots(self, session: SessionState) -> dict:
        return {k: session.slots.get(k) for k in SLOT_KEYS if session.slots.get(k) is not None}

    def _validate_business_slot(self, dt: datetime) -> Optional[str]:
        """Returns an error message if dt is not bookable, else None."""
        now = self.now_fn()
        if dt <= now:
            return "That date and time has already passed. What time would you like instead?"
        if not (config.BUSINESS_HOURS_START <= dt.hour < config.BUSINESS_HOURS_END):
            return (
                f"That's outside our business hours of {config.BUSINESS_HOURS_START}:00 to "
                f"{config.BUSINESS_HOURS_END}:00. What other time works for you?"
            )
        return None

    def handle_turn(self, session: SessionState, utterance: str) -> TurnOutcome:
        session.turn_count += 1

        if session.turn_count > config.MAX_TURN_COUNT:
            session.outcome = "handoff"
            self.session_store.save(session)
            return TurnOutcome(reply=HANDOFF_MESSAGE, end_call=True, outcome="handoff")

        transcript = self.session_store.get_transcript(session.call_sid)[-12:]
        history = [{"speaker": t["speaker"], "text": t["text"]} for t in transcript]
        ctx = TurnContext(
            caller_number=session.caller_number,
            utterance=utterance,
            history=history,
            known_slots=self._clean_slots(session),
            current_action=session.action,
            today=self.now_fn().date(),
        )

        try:
            result = self.intent_engine.parse_turn(ctx)
        except IntentEngineError as exc:
            logger.error("Intent engine failed for call %s: %s", session.call_sid, exc)
            session.outcome = "handoff"
            self.session_store.save(session)
            return TurnOutcome(reply=TECH_FAILURE_MESSAGE, end_call=True, outcome="handoff")

        if result.intent == "unclear":
            session.retry_count += 1
            if session.retry_count > config.MAX_RETRY_COUNT:
                session.outcome = "handoff"
                self.session_store.save(session)
                return TurnOutcome(reply=HANDOFF_MESSAGE, end_call=True, outcome="handoff")
            self.session_store.save(session)
            return TurnOutcome(reply=result.reply or REPEAT_MESSAGE, end_call=False)

        session.retry_count = 0

        if result.intent == "goodbye":
            if session.outcome is None:
                session.outcome = "abandoned"
            self.session_store.save(session)
            return TurnOutcome(reply=result.reply, end_call=True, outcome=session.outcome)

        # result.intent is one of book/reschedule/cancel from here on.
        if session.action is None:
            session.action = result.intent
        elif session.action != result.intent:
            # Caller changed their mind mid-call: start this new action clean.
            session.action = result.intent
            session.slots = {}
            session.target_appointment_id = None

        for key in ("name", "phone", "date_iso", "time_24h", "duration_minutes"):
            value = getattr(result, key)
            if value is not None:
                session.slots[key] = value
        if result.confirmed is not None:
            session.slots["confirmed"] = result.confirmed

        if not session.slots.get("phone"):
            session.slots["phone"] = session.caller_number

        if session.action == "book":
            outcome = self._handle_book(session, result.reply)
        elif session.action == "cancel":
            outcome = self._handle_cancel(session, result.reply)
        else:
            outcome = self._handle_reschedule(session, result.reply)

        self.session_store.save(session)
        return outcome

    def _handle_book(self, session: SessionState, model_reply: str) -> TurnOutcome:
        slots = session.slots
        missing = [k for k in ("name", "phone", "date_iso", "time_24h") if not slots.get(k)]
        if missing:
            return TurnOutcome(reply=model_reply, end_call=False)

        if slots.get("confirmed") is not True:
            return TurnOutcome(reply=model_reply, end_call=False)

        dt = _parse_slot_datetime(slots.get("date_iso"), slots.get("time_24h"))
        if dt is None:
            slots["time_24h"] = None
            slots["confirmed"] = None
            return TurnOutcome(reply="What date and time would you like?", end_call=False)

        validation_error = self._validate_business_slot(dt)
        if validation_error:
            slots["time_24h"] = None
            slots["confirmed"] = None
            return TurnOutcome(reply=validation_error, end_call=False)

        duration = slots.get("duration_minutes") or config.DEFAULT_APPOINTMENT_DURATION_MINUTES
        try:
            appt = self.calendar_store.book(slots["name"], slots["phone"], dt, duration)
        except ConflictError as exc:
            slots["time_24h"] = None
            slots["confirmed"] = None
            return TurnOutcome(
                reply=(
                    f"Sorry, {exc.conflicting.start_time.strftime('%A %B %-d at %-I:%M %p')} "
                    "is already booked. What other time works for you?"
                ),
                end_call=False,
            )

        session.outcome = "booked"
        session.target_appointment_id = appt.id
        session.action = None
        session.slots = {"name": slots["name"], "phone": slots["phone"]}
        reply = (
            f"You're all set, {appt.name}. Your appointment is booked for "
            f"{appt.start_time.strftime('%A %B %-d at %-I:%M %p')}. Is there anything else "
            "I can help with?"
        )
        return TurnOutcome(reply=reply, end_call=False, outcome="booked")

    def _find_target_appointment(self, session: SessionState):
        phone = session.slots.get("phone") or session.caller_number
        matches = self.calendar_store.list_by_phone(phone, active_only=True)
        return matches[0] if matches else None

    def _handle_cancel(self, session: SessionState, model_reply: str) -> TurnOutcome:
        slots = session.slots
        if session.target_appointment_id is None:
            appt = self._find_target_appointment(session)
            if appt is None:
                session.action = None
                session.slots = {}
                return TurnOutcome(
                    reply=(
                        "I don't see any upcoming appointment under that number. Would you "
                        "like to book one instead?"
                    ),
                    end_call=False,
                )
            session.target_appointment_id = appt.id
            when = appt.start_time.strftime("%A %B %-d at %-I:%M %p")
            return TurnOutcome(
                reply=f"I found your appointment for {when}. Should I go ahead and cancel it?",
                end_call=False,
            )

        if slots.get("confirmed") is True:
            try:
                appt = self.calendar_store.cancel(session.target_appointment_id)
            except NotFoundError:
                session.action = None
                session.target_appointment_id = None
                session.slots = {}
                return TurnOutcome(
                    reply="I couldn't find that appointment anymore. Anything else I can help with?",
                    end_call=False,
                )
            session.outcome = "cancelled"
            session.action = None
            session.target_appointment_id = None
            when = appt.start_time.strftime("%A %B %-d at %-I:%M %p")
            session.slots = {}
            return TurnOutcome(
                reply=f"Done, your appointment for {when} is cancelled. Anything else I can help with?",
                end_call=False,
                outcome="cancelled",
            )

        if slots.get("confirmed") is False:
            session.action = None
            session.target_appointment_id = None
            session.slots = {}
            return TurnOutcome(
                reply="Okay, I won't cancel it. Is there anything else I can help with?",
                end_call=False,
            )

        # Confirmation still pending; re-ask.
        appt = self.calendar_store.get(session.target_appointment_id)
        when = appt.start_time.strftime("%A %B %-d at %-I:%M %p")
        return TurnOutcome(
            reply=f"Just to confirm, should I cancel your appointment for {when}?", end_call=False
        )

    def _handle_reschedule(self, session: SessionState, model_reply: str) -> TurnOutcome:
        slots = session.slots
        if session.target_appointment_id is None:
            appt = self._find_target_appointment(session)
            if appt is None:
                session.action = None
                session.slots = {}
                return TurnOutcome(
                    reply=(
                        "I don't see any upcoming appointment under that number. Would you "
                        "like to book a new one instead?"
                    ),
                    end_call=False,
                )
            session.target_appointment_id = appt.id

        if not slots.get("date_iso") or not slots.get("time_24h"):
            return TurnOutcome(reply=model_reply, end_call=False)

        if slots.get("confirmed") is not True:
            old_appt = self.calendar_store.get(session.target_appointment_id)
            new_dt = _parse_slot_datetime(slots.get("date_iso"), slots.get("time_24h"))
            old_when = old_appt.start_time.strftime("%A %B %-d at %-I:%M %p")
            new_when = new_dt.strftime("%A %B %-d at %-I:%M %p") if new_dt else "that time"
            return TurnOutcome(
                reply=f"I'll move your appointment from {old_when} to {new_when}. Shall I confirm that?",
                end_call=False,
            )

        dt = _parse_slot_datetime(slots.get("date_iso"), slots.get("time_24h"))
        if dt is None:
            slots["time_24h"] = None
            slots["confirmed"] = None
            return TurnOutcome(reply="What date and time would you like instead?", end_call=False)

        validation_error = self._validate_business_slot(dt)
        if validation_error:
            slots["time_24h"] = None
            slots["confirmed"] = None
            return TurnOutcome(reply=validation_error, end_call=False)

        duration = slots.get("duration_minutes")
        try:
            appt = self.calendar_store.reschedule(session.target_appointment_id, dt, duration)
        except ConflictError as exc:
            slots["time_24h"] = None
            slots["confirmed"] = None
            return TurnOutcome(
                reply=(
                    f"Sorry, {exc.conflicting.start_time.strftime('%A %B %-d at %-I:%M %p')} "
                    "is already booked. What other time works for you?"
                ),
                end_call=False,
            )
        except NotFoundError:
            session.action = None
            session.target_appointment_id = None
            session.slots = {}
            return TurnOutcome(
                reply="I couldn't find that appointment anymore. Anything else I can help with?",
                end_call=False,
            )

        session.outcome = "rescheduled"
        session.action = None
        session.target_appointment_id = None
        session.slots = {}
        when = appt.start_time.strftime("%A %B %-d at %-I:%M %p")
        return TurnOutcome(
            reply=f"Done, your appointment is moved to {when}. Anything else I can help with?",
            end_call=False,
            outcome="rescheduled",
        )
