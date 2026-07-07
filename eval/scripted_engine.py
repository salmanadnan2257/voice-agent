"""ScriptedIntentEngine: a deterministic, regex-based stand-in for the real
Gemini intent engine (voice_agent.intent_engine.GeminiIntentEngine).

This is a TEST DOUBLE, not a production fallback. It exists so the 20+
scripted scenarios in eval/scenarios.py and most of the pytest suite can run
free, instantly, and deterministically, exercising every bit of the real
conversation state machine (voice_agent/conversation.py) and the real SQLite
calendar store without spending real Gemini API calls on every scenario
turn. The production app (voice_agent/app.py) always uses the real
GeminiIntentEngine; a small, separately marked set of tests
(tests/test_llm_intent.py) proves that real engine works.

Because scenario scripts are hand-written for this engine, its natural-
language coverage only needs to be as good as the phrases the scenarios use.
It is intentionally simple regex/keyword matching, not a general NLU system.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from voice_agent.intent_engine import IntentEngine, TurnContext, TurnResult

_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_BOOK_KW = re.compile(r"\b(book|schedule|set up|new appointment|make an appointment|need an appointment|get an appointment)\b", re.I)
_RESCHEDULE_KW = re.compile(r"\b(reschedul\w*|move my appointment|move it|change my appointment|different time)\b", re.I)
_CANCEL_KW = re.compile(r"\b(cancel|call off|delete my appointment)\b", re.I)
_GOODBYE_KW = re.compile(r"\b(bye|goodbye|that'?s all|nothing else|no thanks|no,? that'?s it|no more)\b", re.I)
_YES_KW = re.compile(r"\b(yes|yeah|yep|yup|correct|that'?s right|sounds good|confirm(?:ed)?|please do|go ahead)\b", re.I)
_NO_KW = re.compile(r"\b(no|nope|wrong|not correct|don'?t|do not)\b", re.I)
_VAGUE_TIME_RE = re.compile(r"\b(morning|afternoon|evening|sometime|later|whenever)\b", re.I)

_NAME_RE = re.compile(r"\b(?:my name is|this is|name'?s|i'?m|i am)\s+([A-Za-z][A-Za-z .'-]{1,40})", re.I)
_PHONE_RE = re.compile(r"(\+?\d[\d\-. ]{6,}\d)")
_TIME_AMPM_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", re.I)
_TIME_24H_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")
_MONTH_DAY_RE = re.compile(
    r"\b(" + "|".join(_MONTHS.keys()) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b", re.I
)
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})\b")
_DURATION_HOUR_RE = re.compile(r"\b(one|1|two|2)\s*hours?\b|\ban hour\b", re.I)
_DURATION_MIN_RE = re.compile(r"\b(\d{1,3})\s*min(?:ute)?s?\b", re.I)
_HALF_HOUR_RE = re.compile(r"\bhalf(?: an)? hour\b|\b30\s*min", re.I)


def _extract_name(utterance: str) -> str | None:
    match = _NAME_RE.search(utterance)
    if not match:
        return None
    raw = match.group(1)
    # Cut off any trailing clause the caller tacked on in the same breath.
    raw = re.split(r",| and | phone| number", raw, maxsplit=1)[0]
    raw = raw.strip().rstrip(".").strip()
    return raw.title() or None


def _extract_phone(utterance: str) -> str | None:
    match = _PHONE_RE.search(utterance)
    if not match:
        return None
    digits = re.sub(r"\D", "", match.group(1))
    return digits if len(digits) >= 7 else None


def _extract_date(utterance: str, today: date) -> str | None:
    low = utterance.lower()
    if "today" in low:
        return today.isoformat()
    if "tomorrow" in low:
        return (today + timedelta(days=1)).isoformat()

    weekday_match = re.search(r"\b(next\s+)?(" + "|".join(_WEEKDAYS.keys()) + r")\b", low)
    if weekday_match:
        forced_next_week = bool(weekday_match.group(1))
        target = _WEEKDAYS[weekday_match.group(2)]
        delta = (target - today.weekday()) % 7
        if delta == 0:
            delta = 7  # same weekday as today almost certainly means the upcoming one, not today
        if forced_next_week:
            delta += 7
        return (today + timedelta(days=delta)).isoformat()

    month_match = _MONTH_DAY_RE.search(low)
    if month_match:
        month = _MONTHS[month_match.group(1).lower()]
        day = int(month_match.group(2))
        year = today.year
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if candidate < today:
            candidate = date(year + 1, month, day)
        return candidate.isoformat()

    numeric_match = _NUMERIC_DATE_RE.search(low)
    if numeric_match:
        month, day = int(numeric_match.group(1)), int(numeric_match.group(2))
        year = today.year
        try:
            candidate = date(year, month, day)
        except ValueError:
            return None
        if candidate < today:
            candidate = date(year + 1, month, day)
        return candidate.isoformat()

    return None


def _extract_time(utterance: str) -> str | None:
    low = utterance.lower()
    if "noon" in low:
        return "12:00"
    if "midnight" in low:
        return "00:00"
    match = _TIME_AMPM_RE.search(low)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        meridiem = match.group(3)
        if meridiem == "pm" and hour != 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"
    match = _TIME_24H_RE.search(low)
    if match:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    return None


def _extract_duration(utterance: str) -> int | None:
    low = utterance.lower()
    if _HALF_HOUR_RE.search(low):
        return 30
    if _DURATION_HOUR_RE.search(low):
        return 60
    match = _DURATION_MIN_RE.search(low)
    if match:
        return int(match.group(1))
    return None


class ScriptedIntentEngine(IntentEngine):
    """Deterministic rule-based test double. See module docstring."""

    def parse_turn(self, ctx: TurnContext) -> TurnResult:
        utterance = ctx.utterance or ""

        new_intent = None
        if _CANCEL_KW.search(utterance):
            new_intent = "cancel"
        elif _RESCHEDULE_KW.search(utterance):
            new_intent = "reschedule"
        elif _BOOK_KW.search(utterance):
            new_intent = "book"
        elif _GOODBYE_KW.search(utterance):
            new_intent = "goodbye"

        name = _extract_name(utterance)
        phone = _extract_phone(utterance)
        date_iso = _extract_date(utterance, ctx.today)
        time_24h = _extract_time(utterance)
        duration_minutes = _extract_duration(utterance)

        confirmed = None
        has_yes = bool(_YES_KW.search(utterance))
        has_no = bool(_NO_KW.search(utterance))
        if has_yes and not has_no:
            confirmed = True
        elif has_no and not has_yes:
            confirmed = False

        extracted_anything = any(
            v is not None for v in (name, phone, date_iso, time_24h, duration_minutes, confirmed)
        )
        vague_answer = bool(_VAGUE_TIME_RE.search(utterance))

        if new_intent is not None:
            intent = new_intent
        elif ctx.current_action is not None and (extracted_anything or vague_answer):
            # Continuing an already-established action: the utterance either
            # yielded a usable slot value, or is a recognizably vague (but
            # real) answer like "sometime in the morning" that warrants a
            # clarifying re-ask rather than a generic "didn't catch that".
            intent = ctx.current_action
        else:
            # No established action to fall back on, no recognized keyword,
            # no usable slot, no recognizable vague answer, and/or no speech
            # at all (Twilio sends "" when it hears nothing): genuinely
            # unclear, which is what drives the retry-then-handoff path.
            intent = "unclear"

        combined = dict(ctx.known_slots)
        for key, value in (
            ("name", name), ("phone", phone), ("date_iso", date_iso),
            ("time_24h", time_24h), ("duration_minutes", duration_minutes),
        ):
            if value is not None:
                combined[key] = value
        if confirmed is not None:
            combined["confirmed"] = confirmed

        reply = _build_reply(intent, combined)

        return TurnResult(
            reply=reply,
            intent=intent,
            name=name,
            phone=phone,
            date_iso=date_iso,
            time_24h=time_24h,
            duration_minutes=duration_minutes,
            confirmed=confirmed,
        )


def _build_reply(intent: str, slots: dict) -> str:
    if intent == "unclear":
        return "Sorry, I didn't catch that. Could you say that again?"
    if intent == "goodbye":
        return "Thanks for calling, goodbye!"
    if intent == "book":
        if not slots.get("name"):
            return "Sure, I can help you book an appointment. Can I get your name please?"
        if not slots.get("phone"):
            return "What phone number should we use for this appointment?"
        if not slots.get("date_iso") or not slots.get("time_24h"):
            return "What date and time would you like to come in?"
        if slots.get("confirmed") is not True:
            return (
                f"So that's {slots['name']} on {slots['date_iso']} at {slots['time_24h']}. "
                "Should I go ahead and book it?"
            )
        return "Great, let me get that booked."
    if intent == "reschedule":
        if not slots.get("date_iso") or not slots.get("time_24h"):
            return "What date and time would you like to move it to?"
        return "Let me pull that up."
    if intent == "cancel":
        return "Let me pull that up."
    return "Could you tell me more?"
