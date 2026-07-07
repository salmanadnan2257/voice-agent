# Voice Agent

A phone-in appointment scheduling agent: Twilio answers the call, Gemini 2.5 Flash
figures out what the caller wants turn by turn, and a local SQLite calendar actually
books, reschedules, and cancels appointments with real conflict detection.

## Why it exists

Small service businesses (clinics, salons, repair shops) lose bookings to missed calls
and voicemail tag. This is a from-scratch demonstration of the pattern that fixes that:
a real telephony front end (Twilio), an LLM doing multi-turn slot-filling instead of a
rigid phone-tree, and a durable calendar backing store with the conflict checks a human
receptionist would do by hand. It is also a concrete example of testing an LLM-driven
system on a budget: the conversation logic is fully covered by 25 scripted scenarios and
most of the pytest suite without spending a cent on the real model, and only a small,
clearly marked slice of tests touches the live, billed Gemini and Twilio APIs.

## Features

- FastAPI endpoints Twilio calls directly for TwiML: `/voice/incoming`, `/voice/gather`,
  `/voice/status`, plus `/health`.
- Multi-turn conversation state machine for three actions: book, reschedule, cancel.
  State is kept per call (keyed by Twilio's CallSid) in SQLite, so it survives the
  separate HTTP request each `<Gather>` round trip makes.
- Gemini 2.5 Flash (via Vertex AI) does intent classification and slot extraction
  (name, phone, date, time, duration, yes/no confirmation) from free-form speech, with
  thinking disabled and a strict JSON response schema so replies are fast and parseable.
- Real SQLite calendar store: conflict detection uses half-open interval overlap (two
  appointments that just touch at the boundary do not conflict), book/reschedule/cancel
  all actually mutate it. An optional Google Calendar adapter (events.insert/patch/delete
  plus freebusy.query for conflicts) sits behind the same interface, used automatically
  if `GOOGLE_CALENDAR_CREDENTIALS_PATH` is set; SQLite is the default and what every test
  in this repo actually exercises.
- Safety net: unclear speech gets a bounded number of "could you repeat that?" retries
  before a human-handoff message and hangup; a hard cap on total turns catches any other
  stuck conversation; any Gemini API failure hands off immediately with a clear message
  instead of leaving the caller stuck. Every call's full transcript and final outcome
  (booked/rescheduled/cancelled/handoff/abandoned) is logged to SQLite.
- Outbound test-call CLI (`voice_agent/cli.py`): places a real Twilio call to a number
  passed explicitly on the command line. Never runs automatically; requires an explicit
  confirmation prompt (or `--yes`) before it dials.
- 25 scripted evaluation scenarios (`eval/scenarios.py`) covering book, reschedule,
  cancel, ambiguous time/date, double-booking conflicts, unclear speech, retry-cap and
  turn-cap handoffs, and abandoned calls, run against the real state machine and a real
  temporary SQLite calendar.

## Architecture

```
Twilio  --HTTP POST-->  voice_agent/app.py  (FastAPI: TwiML endpoints)
                              |
                              v
                    voice_agent/conversation.py   (state machine: slot filling,
                              |                     confirmation, business hours,
                              |                     retries, handoff)
                 ______________|______________
                |                             |
                v                             v
   voice_agent/intent_engine.py     voice_agent/calendar_store.py
   (IntentEngine interface)         (CalendarStore interface)
        |                                |
        v                                v
   GeminiIntentEngine               SqliteCalendarStore (default)
   (real Gemini 2.5 Flash,          GoogleCalendarStore (optional,
    Vertex AI, production           behind GOOGLE_CALENDAR_CREDENTIALS_PATH)
    default)

   voice_agent/session_store.py: per-call state + full transcript, SQLite
   voice_agent/twiml.py: <Gather>/<Say>/<Hangup> generation
   voice_agent/cli.py: outbound test-call CLI (Twilio REST API)

   eval/scripted_engine.py: deterministic regex-based IntentEngine, a TEST
   DOUBLE for the real Gemini engine, used by eval/scenarios.py and most of
   the pytest suite so they run free and reproducibly.
```

The conversation manager only ever depends on the `IntentEngine` and `CalendarStore`
interfaces, never on Gemini or SQLite directly. That is what makes it possible to run
the exact same booking/reschedule/cancel logic in three places: the live app (real
Gemini, real SQLite), the pytest TestClient tests (scripted engine, temp SQLite), and
the offline evaluation harness (scripted engine, temp SQLite).

## Setup

```bash
cd voice-agent
python3 -m venv .venv && source .venv/bin/activate   # keep the venv outside the repo if you prefer
pip install -r requirements.txt
cp .env.example .env   # then fill in real Twilio + Vertex AI credentials
```

Required in `.env`: `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_PHONE_NUMBER`,
`GOOGLE_APPLICATION_CREDENTIALS` (a Vertex AI service account JSON path), `GCP_PROJECT`.
Everything else in `.env.example` is optional; see the comments there.

## Usage

Run the server locally:

```bash
uvicorn voice_agent.app:app --reload --port 8000
curl http://localhost:8000/health
```

This boots with the real Gemini engine and a local SQLite file
(`voice-agent/data/voice_agent.db`, created on first run, gitignored). It answers
`/health` and `/voice/incoming` with no public URL needed; you only need a public URL
once you want an actual Twilio phone number to reach it (see below).

Run the pytest suite (fast, free, deterministic part only):

```bash
pytest -m "not real_api"
```

Run the small number of tests that hit the real Gemini/Twilio APIs:

```bash
pytest -m real_api
```

Run the 25 scripted scenarios standalone with a printed pass/fail report:

```bash
python -m eval.run_eval
```

Place a real outbound test call (requires a public URL, see below):

```bash
python -m voice_agent.cli +15551234567 --base-url https://your-tunnel.ngrok-free.app
```

It refuses to run without a public URL, validates the number looks like E.164, and asks
for an explicit `y` confirmation (or pass `--yes`) before it actually dials, since this
places a real, billed call.

### Getting a real inbound phone-call demo

Nothing in this repo can make an actual phone ring without a URL Twilio can reach over
the public internet, and none is configured right now (the owner said it wasn't needed
yet). To get a real recorded call:

1. Run the server: `uvicorn voice_agent.app:app --port 8000`.
2. In another terminal: `ngrok http 8000` (or deploy this somewhere with a real URL).
3. Put that URL in `.env` as `PUBLIC_WEBHOOK_BASE_URL` (e.g.
   `https://abcd1234.ngrok-free.app`).
4. Either call the Twilio number directly, or run
   `python -m voice_agent.cli +1YOURCELLNUMBER --base-url https://abcd1234.ngrok-free.app`
   to have it call you.
5. In the Twilio Console, point the phone number's "A call comes in" webhook at
   `PUBLIC_WEBHOOK_BASE_URL/voice/incoming` if you want inbound calls to work without the
   CLI.

## Testing and verification actually performed

- **Calendar store**: 13 pytest tests, no network, covering booking, exact and partial
  overlap conflicts, back-to-back non-overlapping slots, reschedule (including excluding
  its own slot from the conflict check), cancel, not-found errors, and persistence across
  reconnects.
- **TwiML/FastAPI endpoints**: 7 pytest tests via FastAPI's TestClient (the scripted
  engine stands in for Gemini; no real Twilio calls happen in either direction), covering
  `/health`, `/voice/incoming`, a full multi-turn booking flow through `/voice/gather`
  that actually updates the calendar, unclear-speech handling, goodbye/hangup, and the
  `/voice/status` abandoned-call path.
- **25 scripted evaluation scenarios**, run for real against the actual conversation
  state machine and a real (temporary) SQLite calendar: book (one-shot, multi-turn,
  explicit duration, explicit phone override, rejected-then-corrected confirmation),
  reschedule (success, not found, conflict-then-alternate), cancel (success, not found),
  double-booking and overlapping/adjacent conflicts, ambiguous time and date, unclear
  speech (recovers, and exceeds the retry cap into handoff), an abandoned call detected
  via the Twilio status callback, business-hours and past-date rejection, a mid-call
  change of mind, two independent concurrent bookings, goodbye-after-success not
  clobbering the outcome, and the absolute turn-count safety cap. **Result: 25/25
  passed.** I deliberately broke the retry cap and turn cap in a throwaway run to confirm
  the two scenarios that check them actually fail when the invariant is broken, rather
  than passing vacuously; both failed as expected, then I reverted the sabotage.
- **A small number of real Gemini calls** (6, in `tests/test_llm_intent.py`, marked
  `real_api`) prove the actual production `GeminiIntentEngine` classifies book/reschedule
  /cancel/unclear correctly and reads back yes/no confirmations, against the live Vertex
  AI credential. One real Twilio call (also marked `real_api`) does a read-only lookup of
  the account's own phone number to confirm it is active and voice-capable; Twilio does
  not bill for read-only API calls. **No real phone call was placed anywhere in this
  repo's tests or its build**, per the project's budget rule: no public webhook URL is
  configured, so a real call could not have completed its TwiML fetch anyway, and Twilio
  integration is instead proven through webhook-signature validation with the real
  auth token (a local HMAC check, no network call) and that one read-only account lookup.
- **Full suite**: 54 tests collected, 54 passed, 0 failed (47 run together with
  `-m "not real_api"`; the other 7, marked `real_api`, run separately by design to keep
  real API spend visible and deliberate).
- `requirements.txt` installs cleanly into a brand new virtualenv (verified in a
  scratchpad venv, not one that had anything pre-installed).
- The real production app (real Gemini engine, real SQLite, no fakes) was booted with
  `uvicorn` and answered `/health` and `/voice/incoming` for real over HTTP.
- **A real public tunnel, end to end, after the initial build**: the app was booted for
  real (real `GeminiIntentEngine`, real SQLite, `enforce_signature=True`) and exposed
  over the public internet via an SSH tunnel (`localhost.run`, no account needed). Three
  requests, shaped exactly like Twilio's real webhook calls and signed with a genuine
  HMAC-SHA1 signature computed from the real `TWILIO_AUTH_TOKEN` (the same algorithm
  Twilio itself uses, not a mock), were sent to the public URL: an opening call, a
  natural-language booking request ("I would like to book an appointment for tomorrow at
  2 PM, my name is Test Caller"), a phone number, and a confirmation. Signature validation
  passed on every request. Gemini correctly resolved "tomorrow" against the real current
  date, extracted the name, time, and phone number across turns, and the conversation
  ended with a real row written to `data/voice_agent.db` (`Test Caller`, `+15551234567`,
  2026-07-08T14:00, 30 minutes, `booked`), confirmed by querying the database directly,
  not by trusting the TwiML response alone. The test row was deleted afterward, the
  tunnel and app were stopped, and `PUBLIC_WEBHOOK_BASE_URL` was reset to blank. The
  live phone number's actual production webhook (already pointed at a separate Vapi
  integration) was never touched; the signature was constructed independently rather
  than by redirecting the real number, specifically to avoid disrupting that existing
  configuration. This closes the one gap the "What I'd do differently" section used to
  list: the signature check and the full multi-turn flow are now proven against a real
  public tunnel, not just unit-tested locally. The one thing still unverified is an
  actual human voice call, since that needs a live tunnel kept up and a real phone in
  the loop, both outside a single verification pass.

### Real API cost incurred

Gemini: 12 real `generateContent` calls total (9 from the initial build: 2 early
exploratory calls, 6 in the marked test file, 1 instrumented measurement; plus 3 more
from the live public-tunnel verification above, one per conversational turn). Same
rate basis as before (roughly $0.30 per million input tokens, $2.50 per million output
tokens for Gemini 2.5 Flash), the total across all 12 calls is still under a tenth of a
cent, an estimate from per-token rates, not a verified line-item invoice. Twilio: one
real read-only REST API call (phone number lookup) during the build, plus one more
during the tunnel verification (fetching the number's current webhook config, also
read-only) and one lookup of the phone number's SID; Twilio does not charge for any of
these. No call, SMS, or paid Lookup API was used anywhere, so Twilio's real cost
incurred is $0.00. No changes were made to the live phone number's configuration.

## Challenges

1. **Twilio does not call your action URL on silence by default.** `<Gather>` only
   requests its `action` URL when it captures speech; on a silent timeout it just moves
   to the next TwiML verb, which would have made "unclear speech" and the retry/handoff
   logic in `conversation.py` unreachable from a real silent caller. Fixed by setting
   `actionOnEmptyResult="true"` on every `<Gather>` in `voice_agent/twiml.py`, so every
   outcome, speech or silence, comes back through `/voice/gather` and the same state
   machine handles it.
2. **Testing an LLM-driven system on a real budget.** Twenty-five scripted scenarios,
   several with 5+ turns, would be 100+ real Gemini calls if run against the actual
   model. Solved by defining an `IntentEngine` interface (`voice_agent/intent_engine.py`)
   with two implementations: the real `GeminiIntentEngine` (production default) and a
   deterministic regex-based `ScriptedIntentEngine` (`eval/scripted_engine.py`), used
   only by tests and the evaluation harness and documented in its own docstring as a test
   double, not a production fallback, to keep the "LLM-driven" claim honest.
3. **Relative dates and business-hours checks broke test determinism.** The conversation
   manager originally called `datetime.now()` directly for both the turn context's "today"
   and the past/business-hours validation, so a scenario asserting an exact resulting
   timestamp would only be correct on the day it happened to run. Fixed by threading a
   single injectable `now_fn` through `ConversationManager` into both `TurnContext.today`
   and the validation check, and pinning every scripted scenario to one fixed reference
   time (`eval/scenarios.py`, `FIXED_NOW = datetime(2026, 7, 6, 13, 0)`).
4. **Scenarios that looked right on paper didn't exercise the code path they were named
   for.** The first draft of the business-hours and past-date rejection scenarios had the
   caller correct a bad time before ever confirming it, so `_validate_business_slot`
   (which only runs after `confirmed is True`) never actually ran; the scenario passed
   for the wrong reason. Caught by tracing the exact turn-by-turn state transitions
   before the first eval run, and fixed by adding an explicit "yes" turn that confirms
   the bad time first, forcing the rejection path to actually fire.
5. **Gemini's default thinking mode is too slow and too expensive for a live phone
   turn.** An early exploratory call with default settings took noticeably longer and
   spent 56 thinking tokens on a one-word answer. Measured a second call with
   `thinking_config=ThinkingConfig(thinking_budget=0)` plus a strict `response_schema`
   and `response_mime_type="application/json"`: 1.8 seconds and a clean structured
   response. That configuration is what `GeminiIntentEngine` uses in production, since a
   caller on the phone cannot tolerate multi-second silences per turn.
6. **FastAPI's `url_for` returns an absolute URL, not the relative path I first
   assumed.** A TwiML endpoint test asserted the `<Gather action=...>` attribute equaled
   `/voice/gather` exactly; it actually returns `http://testserver/voice/gather` when no
   `PUBLIC_WEBHOOK_BASE_URL` is configured (the real fallback behavior, not a bug), caught
   by an actual failing pytest run. Fixed the test's assertion, not the code.

## What I learned

- Conflict-free interval overlap is `a_start < b_end and b_start < a_end` (a half-open
  interval check), which correctly allows two appointments to touch at the boundary
  (one ending exactly when the next starts) without a false-positive conflict; verified
  with an explicit boundary test in `tests/test_calendar_store.py`.
- Structuring an LLM call for a latency-sensitive product (a live phone turn) means
  actively turning features off: Gemini 2.5 Flash's default "thinking" adds real seconds
  and real tokens even for trivial prompts, and `response_schema` plus
  `response_mime_type="application/json"` gets a reliably parseable answer without
  needing to also ask the model, in prose, to "please output JSON."
- Designing around an interface (`IntentEngine`, `CalendarStore`) instead of a concrete
  class up front is what actually made a 25-scenario evaluation harness affordable to
  run repeatedly: the harness and most of the test suite never touch the real, billed
  dependency, only the small number of tests that specifically exist to verify it do.
- A test or scenario that passes is not proof it is testing anything: I had to actually
  break two of the safety-cap scenarios' preconditions in a throwaway run to confirm
  they'd fail if the code regressed, rather than trusting the assertion by inspection
  alone.

## What I'd do differently

- Timezones are not handled at all: every date/time is treated as one implicit local
  timezone for both the business and every caller. A real deployment serving callers
  across timezones would need to ask for or infer the caller's timezone and convert.
- The real Gemini intent engine is only proven against clean, hand-typed English
  utterances (`tests/test_llm_intent.py`); it has never been run against actual Twilio
  speech-to-text output, which is noisier (mumbling, background noise, partial phrases).
  That is a real gap between "the API call works" and "this works on a real phone call."
- The Google Calendar adapter (`GoogleCalendarStore` in `voice_agent/calendar_store.py`)
  is implemented fully against the real Calendar API v3, but no Google Calendar
  credentials were provided, so it has never actually been run against a live calendar.
  It is not a stub, but it is not verified end to end either.
- SQLite is a single-writer database; this is fine for one phone line and one process,
  but a deployment with multiple concurrent lines or worker processes would need
  Postgres (or at least WAL mode plus real concurrency testing) instead.
- The webhook-signature check and the full multi-turn flow have now been exercised
  against a real public tunnel (see "Testing and verification actually performed"
  above), but only with hand-typed English text standing in for Twilio's speech
  recognition output. An actual human voice call, with real speech-to-text noise and a
  live tunnel held open for the duration, is still the one thing that needs the owner in
  the loop.
