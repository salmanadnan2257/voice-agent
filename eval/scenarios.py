"""20+ scripted conversation scenarios, run against the real conversation
state machine (voice_agent.conversation.ConversationManager), a real
temporary SQLite calendar store, and a real temporary SQLite session store.
Only the LLM is swapped for the deterministic ScriptedIntentEngine (see
scripted_engine.py) so this runs free and reproducibly, in or out of CI.

Each scenario is one or more simulated phone calls (a caller_number plus a
list of caller utterances fed straight into ConversationManager.handle_turn,
bypassing Twilio and FastAPI entirely) and a `check` function that asserts
on the resulting calendar state and the per-call transcript/outcome.

Run directly: `python -m eval.run_eval`
Run via pytest: `pytest tests/test_eval_scenarios.py`
"""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

from voice_agent.calendar_store import SqliteCalendarStore
from voice_agent.conversation import ConversationManager
from voice_agent.session_store import SessionState, SessionStore

from .scripted_engine import ScriptedIntentEngine

# Fixed reference "now" for every scenario: Monday 2026-07-06, 13:00. Keeps
# relative-date parsing and business-hours/past-date checks fully
# reproducible regardless of the real wall-clock date the suite runs on.
# Chosen mid-afternoon (not right at business-hours open) so a same-day
# "past" time can be constructed that doesn't also trip the business-hours
# check, keeping those two edge cases independently testable.
FIXED_NOW = datetime(2026, 7, 6, 13, 0, 0)


@dataclass
class CallScript:
    caller_number: str
    turns: list[str]
    simulate_status_callback_after: bool = False


@dataclass
class Scenario:
    name: str
    category: str
    calls: list[CallScript]
    check: Callable[["ScenarioResult"], tuple[bool, str]]


@dataclass
class ScenarioResult:
    calendar_store: SqliteCalendarStore
    session_store: SessionStore
    sessions: dict[str, SessionState] = field(default_factory=dict)

    def session_for(self, call_index: int, scenario_name: str) -> SessionState:
        return self.sessions[f"{scenario_name}-call{call_index}"]


def run_scenario(scenario: Scenario) -> tuple[bool, str, ScenarioResult]:
    tmp_dir = tempfile.mkdtemp(prefix="voice_agent_eval_")
    try:
        calendar_store = SqliteCalendarStore(os.path.join(tmp_dir, "calendar.db"))
        session_store = SessionStore(os.path.join(tmp_dir, "session.db"))
        engine = ScriptedIntentEngine()
        manager = ConversationManager(
            intent_engine=engine,
            calendar_store=calendar_store,
            session_store=session_store,
            now_fn=lambda: FIXED_NOW,
        )

        sessions: dict[str, SessionState] = {}
        for idx, call in enumerate(scenario.calls):
            call_sid = f"{scenario.name}-call{idx}"
            session = session_store.get_or_create(call_sid, call.caller_number)
            session_store.log_turn(call_sid, 0, "agent", "[greeting]")
            for turn_text in call.turns:
                session_store.log_turn(call_sid, session.turn_count, "caller", turn_text or "[no speech detected]")
                outcome = manager.handle_turn(session, turn_text)
                session_store.log_turn(call_sid, session.turn_count, "agent", outcome.reply)
                if outcome.end_call:
                    break
            if call.simulate_status_callback_after:
                session_store.mark_abandoned_if_unresolved(call_sid)
                session = session_store.get_or_create(call_sid, call.caller_number)
            sessions[call_sid] = session

        result = ScenarioResult(calendar_store=calendar_store, session_store=session_store, sessions=sessions)
        try:
            passed, detail = scenario.check(result)
        except Exception as exc:  # a check bug shouldn't crash the whole run
            passed, detail = False, f"scenario check raised {exc!r}"
        calendar_store.close()
        session_store.close()
        return passed, detail, result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _transcript_nonempty(result: ScenarioResult, call_sid: str) -> bool:
    return len(result.session_store.get_transcript(call_sid)) >= 2


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = []


def _register(scenario: Scenario) -> None:
    SCENARIOS.append(scenario)


# 1. Simple one-shot book: everything given in a single utterance.
def _check_1(result: ScenarioResult):
    cs = result.calendar_store
    appts = cs.list_by_phone("+15550000001")
    if len(appts) != 1:
        return False, f"expected 1 appointment, found {len(appts)}"
    appt = appts[0]
    ok = (
        appt.name == "Alice Walker"
        and appt.start_time == datetime(2026, 7, 15, 14, 0)
        and appt.duration_minutes == 30
        and appt.status == "booked"
    )
    session = result.session_for(0, "book_simple_one_shot")
    return (ok and session.outcome == "booked"), f"appt={appt}, outcome={session.outcome}"


_register(
    Scenario(
        name="book_simple_one_shot",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000001",
                turns=[
                    "I'd like to book an appointment. My name is Alice Walker, "
                    "on July 15th at 2pm.",
                    "yes go ahead",
                ],
            )
        ],
        check=_check_1,
    )
)


# 2. Multi-turn slot filling: one piece of information per turn.
def _check_2(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("5552223333")
    if len(appts) != 1:
        return False, f"expected 1 appointment for spoken phone, found {len(appts)}"
    appt = appts[0]
    ok = appt.name == "Bob Diaz" and appt.start_time == datetime(2026, 7, 16, 10, 0)
    session = result.session_for(0, "book_multi_turn_slot_filling")
    return (ok and session.outcome == "booked"), f"appt={appt}"


_register(
    Scenario(
        name="book_multi_turn_slot_filling",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000002",
                turns=[
                    "I want to book an appointment",
                    "My name is Bob Diaz",
                    "555-222-3333",
                    "July 16th at 10am",
                    "yes that's right",
                ],
            )
        ],
        check=_check_2,
    )
)


# 3. Explicit duration (one hour instead of the 30 minute default).
def _check_3(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000003")
    if len(appts) != 1:
        return False, f"expected 1 appointment, found {len(appts)}"
    appt = appts[0]
    ok = appt.duration_minutes == 60 and appt.start_time == datetime(2026, 7, 17, 11, 0)
    return ok, f"appt={appt}"


_register(
    Scenario(
        name="book_explicit_duration_one_hour",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000003",
                turns=[
                    "I'd like to schedule an appointment for one hour.",
                    "My name is Carla Nunez, July 17th at 11am.",
                    "yes confirm",
                ],
            )
        ],
        check=_check_3,
    )
)


# 4. Caller states a phone number explicitly, different from the caller ID
#    Twilio reports: the spoken number should win.
def _check_4(result: ScenarioResult):
    by_spoken = result.calendar_store.list_by_phone("5551112222")
    by_caller_id = result.calendar_store.list_by_phone("+19998887777")
    ok = len(by_spoken) == 1 and len(by_caller_id) == 0
    return ok, f"by_spoken={by_spoken}, by_caller_id={by_caller_id}"


_register(
    Scenario(
        name="book_explicit_phone_override",
        category="book",
        calls=[
            CallScript(
                caller_number="+19998887777",
                turns=[
                    "I'd like to book an appointment. My name is Deshawn Price, "
                    "my phone number is 5551112222, July 15th at 3pm.",
                    "yes",
                ],
            )
        ],
        check=_check_4,
    )
)


# 5. Caller rejects the read-back confirmation once, corrects the time, then confirms.
def _check_5(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000005")
    if len(appts) != 1:
        return False, f"expected exactly 1 appointment (not 2), found {len(appts)}"
    appt = appts[0]
    ok = appt.start_time == datetime(2026, 7, 15, 16, 0)
    return ok, f"appt={appt}"


_register(
    Scenario(
        name="book_reject_confirmation_then_correct",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000005",
                turns=[
                    "I'd like to book an appointment. My name is Ewa Kowalski, "
                    "July 15th at 3pm.",
                    "no that's wrong",
                    "4pm please",
                    "yes that's correct",
                ],
            )
        ],
        check=_check_5,
    )
)


# 6. Reschedule an appointment booked earlier in a prior call from the same number.
def _check_6(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000006")
    if len(appts) != 1:
        return False, f"expected 1 active appointment after reschedule, found {len(appts)}"
    appt = appts[0]
    ok = appt.start_time == datetime(2026, 7, 19, 16, 0)
    session = result.session_for(1, "reschedule_success")
    return (ok and session.outcome == "rescheduled"), f"appt={appt}, outcome={session.outcome}"


_register(
    Scenario(
        name="reschedule_success",
        category="reschedule",
        calls=[
            CallScript(
                caller_number="+15550000006",
                turns=["I'd like to book an appointment. My name is Fiona Grant, July 18th at 3pm.", "yes book it"],
            ),
            CallScript(
                caller_number="+15550000006",
                turns=["I need to reschedule my appointment", "July 19th at 4pm", "yes confirm"],
            ),
        ],
        check=_check_6,
    )
)


# 7. Reschedule requested with no appointment on file: graceful not-found, then goodbye -> abandoned.
def _check_7(result: ScenarioResult):
    session = result.session_for(0, "reschedule_not_found_then_abandoned")
    appts = result.calendar_store.list_by_phone("+15550000007")
    return (session.outcome == "abandoned" and len(appts) == 0), f"outcome={session.outcome}, appts={appts}"


_register(
    Scenario(
        name="reschedule_not_found_then_abandoned",
        category="reschedule",
        calls=[
            CallScript(
                caller_number="+15550000007",
                turns=["I need to reschedule my appointment", "no, never mind, goodbye"],
            )
        ],
        check=_check_7,
    )
)


# 8. Cancel an appointment booked earlier in a prior call.
def _check_8(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000008", active_only=True)
    all_appts = result.calendar_store.list_by_phone("+15550000008", active_only=False)
    session = result.session_for(1, "cancel_success")
    ok = len(appts) == 0 and len(all_appts) == 1 and all_appts[0].status == "cancelled"
    return (ok and session.outcome == "cancelled"), f"active={appts}, all={all_appts}, outcome={session.outcome}"


_register(
    Scenario(
        name="cancel_success",
        category="cancel",
        calls=[
            CallScript(
                caller_number="+15550000008",
                turns=["I'd like to book an appointment. My name is Grace Oyelaran, July 20th at 9am.", "yes"],
            ),
            CallScript(caller_number="+15550000008", turns=["I need to cancel my appointment", "yes cancel it"]),
        ],
        check=_check_8,
    )
)


# 9. Cancel requested with no appointment on file: graceful not-found, then goodbye -> abandoned.
def _check_9(result: ScenarioResult):
    session = result.session_for(0, "cancel_not_found_then_abandoned")
    return (session.outcome == "abandoned"), f"outcome={session.outcome}"


_register(
    Scenario(
        name="cancel_not_found_then_abandoned",
        category="cancel",
        calls=[
            CallScript(
                caller_number="+15550000009",
                turns=["I want to cancel my appointment", "no that's all, bye"],
            )
        ],
        check=_check_9,
    )
)


# 10. Double-booking conflict: second caller wants the same slot as an
#     existing appointment, gets refused, then successfully books another time.
def _check_10(result: ScenarioResult):
    cs = result.calendar_store
    first = cs.list_by_phone("+15550000010")
    second = cs.list_by_phone("+15550000011")
    ok = (
        len(first) == 1
        and first[0].start_time == datetime(2026, 7, 20, 13, 0)
        and len(second) == 1
        and second[0].start_time == datetime(2026, 7, 20, 14, 0)
    )
    session = result.session_for(1, "double_booking_conflict")
    return (ok and session.outcome == "booked"), f"first={first}, second={second}"


_register(
    Scenario(
        name="double_booking_conflict",
        category="conflict",
        calls=[
            CallScript(
                caller_number="+15550000010",
                turns=["I'd like to book an appointment. My name is Hugo Reyes, July 20th at 1pm.", "yes"],
            ),
            CallScript(
                caller_number="+15550000011",
                turns=[
                    "I'd like to book an appointment",
                    "My name is Frank Lee, July 20th at 1pm",
                    "yes go ahead",
                    "2pm works",
                    "yes",
                ],
            ),
        ],
        check=_check_10,
    )
)


# 11. Ambiguous time ("sometime in the morning"), then clarified.
def _check_11(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000012")
    ok = len(appts) == 1 and appts[0].start_time == datetime(2026, 7, 21, 9, 0)
    session = result.session_for(0, "ambiguous_time_then_clarified")
    return (ok and session.turn_count >= 4), f"appts={appts}, turn_count={session.turn_count}"


_register(
    Scenario(
        name="ambiguous_time_then_clarified",
        category="ambiguous_time",
        calls=[
            CallScript(
                caller_number="+15550000012",
                turns=[
                    "I'd like to book an appointment, my name is Ivy Chen",
                    "sometime in the morning",
                    "July 21st at 9am",
                    "yes",
                ],
            )
        ],
        check=_check_11,
    )
)


# 12. Ambiguous date ("later this week"), then clarified with a concrete date.
def _check_12(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000013")
    ok = len(appts) == 1 and appts[0].start_time == datetime(2026, 7, 22, 11, 0)
    session = result.session_for(0, "ambiguous_date_then_clarified")
    return (ok and session.turn_count >= 4), f"appts={appts}, turn_count={session.turn_count}"


_register(
    Scenario(
        name="ambiguous_date_then_clarified",
        category="ambiguous_time",
        calls=[
            CallScript(
                caller_number="+15550000013",
                turns=[
                    "I'd like to book an appointment, my name is Jamal Osei",
                    "later this week at 11am",
                    "July 22nd at 11am",
                    "yes",
                ],
            )
        ],
        check=_check_12,
    )
)


# 13. Unclear speech once, then the caller recovers with a clear answer.
def _check_13(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000014")
    ok = len(appts) == 1 and appts[0].name == "Henry Cole"
    session = result.session_for(0, "unclear_speech_recovers")
    return (ok and session.retry_count == 0 and session.outcome == "booked"), f"appts={appts}"


_register(
    Scenario(
        name="unclear_speech_recovers",
        category="unclear_speech",
        calls=[
            CallScript(
                caller_number="+15550000014",
                turns=[
                    "I'd like to book an appointment",
                    "",  # Twilio heard nothing
                    "My name is Henry Cole, July 22nd at 10am.",
                    "yes",
                ],
            )
        ],
        check=_check_13,
    )
)


# 14. Unclear speech repeated past the retry cap: hands off.
def _check_14(result: ScenarioResult):
    session = result.session_for(0, "unclear_speech_retry_cap_handoff")
    return (session.outcome == "handoff"), f"outcome={session.outcome}, retry_count={session.retry_count}"


_register(
    Scenario(
        name="unclear_speech_retry_cap_handoff",
        category="unclear_speech",
        calls=[
            CallScript(
                caller_number="+15550000015",
                turns=[
                    "I'd like to book an appointment",
                    "mmphf static crackle",
                    "uhh mmphf garble noise",
                    "xk qq zz static",
                    "still nothing useful here",
                ],
            )
        ],
        check=_check_14,
    )
)


# 15. Caller abandons mid-booking; Twilio's call-status callback reports
#     completed before any outcome was reached.
def _check_15(result: ScenarioResult):
    session = result.session_for(0, "abandoned_call_status_callback")
    return (session.outcome == "abandoned"), f"outcome={session.outcome}"


_register(
    Scenario(
        name="abandoned_call_status_callback",
        category="abandoned",
        calls=[
            CallScript(
                caller_number="+15550000016",
                turns=["I'd like to book an appointment", "My name is Karolina Nowak"],
                simulate_status_callback_after=True,
            )
        ],
        check=_check_15,
    )
)


# 16. Requested time before business hours: rejected, then corrected.
def _check_16(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000017")
    ok = len(appts) == 1 and appts[0].start_time == datetime(2026, 7, 23, 9, 0)
    return ok, f"appts={appts}"


_register(
    Scenario(
        name="business_hours_rejection_too_early",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000017",
                turns=[
                    "I'd like to book an appointment. My name is Liam O'Brien, July 23rd at 7am.",
                    "yes",  # confirms the too-early time so the business-hours check actually runs
                    "9am then",
                    "yes",
                ],
            )
        ],
        check=_check_16,
    )
)


# 17. Requested time after business hours: rejected, then corrected.
def _check_17(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000018")
    ok = len(appts) == 1 and appts[0].start_time == datetime(2026, 7, 23, 16, 0)
    return ok, f"appts={appts}"


_register(
    Scenario(
        name="business_hours_rejection_too_late",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000018",
                turns=[
                    "I'd like to book an appointment. My name is Mia Costa, July 23rd at 9pm.",
                    "yes",  # confirms the too-late time so the business-hours check actually runs
                    "4pm instead",
                    "yes",
                ],
            )
        ],
        check=_check_17,
    )
)


# 18. Requested date already in the past: rejected, then corrected to a future date.
def _check_18(result: ScenarioResult):
    appts = result.calendar_store.list_by_phone("+15550000019")
    ok = len(appts) == 1 and appts[0].start_time == datetime(2026, 7, 24, 10, 0)
    return ok, f"appts={appts}"


_register(
    Scenario(
        name="past_date_rejection_then_corrected",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000019",
                turns=[
                    "I'd like to book an appointment. My name is Noah Kim, today at 10am.",
                    "yes",  # confirms the already-passed time so the past-date check actually runs
                    "July 24th at 10am instead",
                    "yes",
                ],
            )
        ],
        check=_check_18,
    )
)


# 19. Two appointments back to back with no gap conflict: 9:00-9:30 and
#     9:30-10:00 must both succeed (touching endpoints are not an overlap).
def _check_19(result: ScenarioResult):
    a = result.calendar_store.list_by_phone("+15550000020")
    b = result.calendar_store.list_by_phone("+15550000021")
    ok = (
        len(a) == 1
        and a[0].start_time == datetime(2026, 7, 25, 9, 0)
        and len(b) == 1
        and b[0].start_time == datetime(2026, 7, 25, 9, 30)
    )
    return ok, f"a={a}, b={b}"


_register(
    Scenario(
        name="boundary_adjacent_appointments_no_conflict",
        category="conflict",
        calls=[
            CallScript(
                caller_number="+15550000020",
                turns=["I'd like to book an appointment. My name is Omar Haddad, July 25th at 9am.", "yes"],
            ),
            CallScript(
                caller_number="+15550000021",
                turns=["I'd like to book an appointment. My name is Priya Nair, July 25th at 9:30am.", "yes"],
            ),
        ],
        check=_check_19,
    )
)


# 20. Overlapping partial conflict: new request starts mid-way through an
#     existing appointment.
def _check_20(result: ScenarioResult):
    a = result.calendar_store.list_by_phone("+15550000022")
    b = result.calendar_store.list_by_phone("+15550000023")
    ok = (
        len(a) == 1
        and a[0].start_time == datetime(2026, 7, 26, 10, 0)
        and len(b) == 1
        and b[0].start_time == datetime(2026, 7, 26, 11, 0)  # pushed to the next free slot offered
    )
    return ok, f"a={a}, b={b}"


_register(
    Scenario(
        name="overlapping_partial_conflict",
        category="conflict",
        calls=[
            CallScript(
                caller_number="+15550000022",
                turns=["I'd like to book an appointment. My name is Quinn Baxter, July 26th at 10am.", "yes"],
            ),
            CallScript(
                caller_number="+15550000023",
                turns=[
                    "I'd like to book an appointment",
                    "My name is Ravi Shah, July 26th at 10:15am",
                    "yes",
                    "11am works",
                    "yes",
                ],
            ),
        ],
        check=_check_20,
    )
)


# 21. Mid-call change of mind: starts a booking, switches to cancel with no
#     appointment on file, then abandons.
def _check_21(result: ScenarioResult):
    session = result.session_for(0, "mid_call_change_of_mind")
    appts = result.calendar_store.list_by_phone("+15550000024")
    return (session.outcome == "abandoned" and len(appts) == 0), f"outcome={session.outcome}, appts={appts}"


_register(
    Scenario(
        name="mid_call_change_of_mind",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000024",
                turns=[
                    "I'd like to book an appointment. My name is Sofia Almeida.",
                    "actually, cancel my appointment instead",
                    "no never mind, goodbye",
                ],
            )
        ],
        check=_check_21,
    )
)


# 22. Two different callers book non-overlapping times: both succeed independently.
def _check_22(result: ScenarioResult):
    a = result.calendar_store.list_by_phone("+15550000025")
    b = result.calendar_store.list_by_phone("+15550000026")
    ok = len(a) == 1 and len(b) == 1 and a[0].start_time != b[0].start_time
    return ok, f"a={a}, b={b}"


_register(
    Scenario(
        name="multiple_independent_bookings",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000025",
                turns=["I'd like to book an appointment. My name is Tariq Malik, July 27th at 9am.", "yes"],
            ),
            CallScript(
                caller_number="+15550000026",
                turns=["I'd like to book an appointment. My name is Uma Reddy, July 27th at 1pm.", "yes"],
            ),
        ],
        check=_check_22,
    )
)


# 23. Goodbye right after a successful booking must not overwrite the outcome.
def _check_23(result: ScenarioResult):
    session = result.session_for(0, "goodbye_after_success_keeps_outcome")
    appts = result.calendar_store.list_by_phone("+15550000027")
    return (session.outcome == "booked" and len(appts) == 1), f"outcome={session.outcome}"


_register(
    Scenario(
        name="goodbye_after_success_keeps_outcome",
        category="book",
        calls=[
            CallScript(
                caller_number="+15550000027",
                turns=[
                    "I'd like to book an appointment. My name is Victor Huang, July 28th at 2pm.",
                    "yes",
                    "no that's all, goodbye",
                ],
            )
        ],
        check=_check_23,
    )
)


# 24. The overall turn cap forces a handoff even without repeated unclear
#     speech: the caller keeps giving a recognizable-but-vague answer
#     ("sometime in the morning") forever, so retry_count never climbs (each
#     turn is a valid continuation, just never a complete one), and only the
#     absolute turn-count safety net ends the call.
def _check_24(result: ScenarioResult):
    session = result.session_for(0, "max_turn_cap_handoff")
    return (
        session.outcome == "handoff" and session.retry_count == 0
    ), f"outcome={session.outcome}, turn_count={session.turn_count}, retry_count={session.retry_count}"


_register(
    Scenario(
        name="max_turn_cap_handoff",
        category="unclear_speech",
        calls=[
            CallScript(
                caller_number="+15550000028",
                turns=["I'd like to book an appointment. My name is Wendy Farrow."]
                + ["sometime in the morning"] * 14,
            )
        ],
        check=_check_24,
    )
)


# 25. Rescheduling into a slot that conflicts with someone else's appointment,
#     then choosing an alternate time.
def _check_25(result: ScenarioResult):
    existing = result.calendar_store.list_by_phone("+15550000029")
    moved = result.calendar_store.list_by_phone("+15550000030")
    ok = (
        len(existing) == 1
        and existing[0].start_time == datetime(2026, 7, 29, 10, 0)
        and len(moved) == 1
        and moved[0].start_time == datetime(2026, 7, 29, 11, 0)
    )
    session = result.session_for(1, "reschedule_conflict_then_alternate")
    return (ok and session.outcome == "rescheduled"), f"existing={existing}, moved={moved}"


_register(
    Scenario(
        name="reschedule_conflict_then_alternate",
        category="conflict",
        calls=[
            CallScript(
                caller_number="+15550000029",
                turns=["I'd like to book an appointment. My name is Xander Cole, July 29th at 10am.", "yes"],
            ),
            CallScript(
                caller_number="+15550000030",
                turns=[
                    "I'd like to book an appointment. My name is Yara Haidari, July 29th at 3pm.",
                    "yes",
                    "I need to reschedule my appointment",
                    "July 29th at 10am",
                    "yes confirm",
                    "11am then",
                    "yes",
                ],
            ),
        ],
        check=_check_25,
    )
)


def run_all() -> list[tuple[str, str, bool, str]]:
    """Runs every registered scenario. Returns (name, category, passed, detail) tuples."""
    results = []
    for scenario in SCENARIOS:
        passed, detail, _ = run_scenario(scenario)
        results.append((scenario.name, scenario.category, passed, detail))
    return results
