"""The LLM-driven intent handler: turns one caller utterance into structured
booking data using Gemini 2.5 Flash on Vertex AI.

``IntentEngine`` is the interface the rest of the app (conversation.py) talks
to. ``GeminiIntentEngine`` is the real, production implementation and is what
app.py wires up by default. Tests and the offline evaluation harness
(eval/scenarios.py) use a separate deterministic stand-in
(``eval.scripted_engine.ScriptedIntentEngine``) so the 20+ scripted scenarios
and most of the pytest suite run free and fully deterministic; a small,
clearly marked handful of tests exercise this real Gemini engine end to end.
"""
from __future__ import annotations

import abc
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger("voice_agent.intent_engine")

VALID_INTENTS = {"book", "reschedule", "cancel", "unclear", "goodbye"}


@dataclass
class TurnContext:
    """Everything the intent engine needs to interpret one caller turn."""

    caller_number: str
    utterance: str  # "" if Twilio's <Gather> captured no speech
    history: list[dict] = field(default_factory=list)  # [{"speaker": "agent"/"caller", "text": ...}]
    known_slots: dict = field(default_factory=dict)  # name/phone/date/time/duration_minutes so far
    current_action: Optional[str] = None  # book/reschedule/cancel, if already determined
    today: date = field(default_factory=date.today)


@dataclass
class TurnResult:
    reply: str
    intent: str  # one of VALID_INTENTS
    name: Optional[str] = None
    phone: Optional[str] = None
    date_iso: Optional[str] = None  # "YYYY-MM-DD"
    time_24h: Optional[str] = None  # "HH:MM"
    duration_minutes: Optional[int] = None
    confirmed: Optional[bool] = None


class IntentEngineError(RuntimeError):
    """Raised when the engine cannot produce a usable result (e.g. API failure)."""


class IntentEngine(abc.ABC):
    @abc.abstractmethod
    def parse_turn(self, ctx: TurnContext) -> TurnResult:
        ...


SYSTEM_PROMPT_TEMPLATE = """You are a phone appointment scheduling assistant for a small \
business. You are having a live, spoken phone conversation, one turn at a time. Today's \
date is {today}. Business hours are 9:00 to 17:00, Monday through Sunday, in the \
business's local time. Default appointment duration is 30 minutes unless the caller asks \
for something else.

Your job each turn: read the conversation so far and the caller's latest utterance, then:
1. Determine the caller's intent: "book" (new appointment), "reschedule" (move an existing \
one), "cancel" (cancel an existing one), "goodbye" (caller wants to end the call, e.g. \
after a completed action or explicitly says bye/no more help needed), or "unclear" (you \
cannot tell what they want, or they gave no usable speech).
2. Extract any of these details present so far IN THE WHOLE CONVERSATION, not just the \
last turn (carry forward anything already known that the caller has not changed): \
caller's name, phone number (digits), appointment date (resolve relative dates like \
"tomorrow" or "next Tuesday" against today's date, output as YYYY-MM-DD), appointment \
time (resolve to 24-hour HH:MM), duration in minutes, and whether the caller has \
explicitly confirmed ("yes that's right") or rejected ("no that's wrong") a summary you \
read back to them. Leave a field null if it is genuinely not yet known.
3. Write a short, natural, spoken-style reply (one or two sentences, no markdown, no \
lists) that either: asks for the next missing piece of information one at a time, reads \
back a full summary and asks for explicit yes/no confirmation once name+date+time are \
all known, acknowledges a completed action, or asks the caller to repeat themselves if \
unclear.

Currently known so far: {known_slots}
Current action already established this call (may be null): {current_action}

Respond with ONLY the JSON object described by the schema. No prose outside the JSON.
"""


def _build_contents(ctx: TurnContext) -> str:
    lines = []
    for turn in ctx.history:
        speaker = "Assistant" if turn.get("speaker") == "agent" else "Caller"
        lines.append(f"{speaker}: {turn.get('text', '')}")
    lines.append(f"Caller: {ctx.utterance!r}" if ctx.utterance else "Caller: [no speech detected]")
    return "\n".join(lines)


class GeminiIntentEngine(IntentEngine):
    """Production intent engine: Gemini 2.5 Flash via Vertex AI (google-genai SDK)."""

    def __init__(self, project: str, location: str = "global", model: str = "gemini-2.5-flash"):
        from google import genai

        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._model = model
        # Running total for cost visibility (e.g. logged at shutdown, or
        # inspected after a batch of calls); not persisted anywhere.
        self.total_prompt_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0

    def _response_schema(self):
        from google.genai import types

        return types.Schema(
            type=types.Type.OBJECT,
            properties={
                "intent": types.Schema(type=types.Type.STRING, enum=sorted(VALID_INTENTS)),
                "reply": types.Schema(type=types.Type.STRING),
                "name": types.Schema(type=types.Type.STRING, nullable=True),
                "phone": types.Schema(type=types.Type.STRING, nullable=True),
                "date_iso": types.Schema(type=types.Type.STRING, nullable=True),
                "time_24h": types.Schema(type=types.Type.STRING, nullable=True),
                "duration_minutes": types.Schema(type=types.Type.INTEGER, nullable=True),
                "confirmed": types.Schema(type=types.Type.BOOLEAN, nullable=True),
            },
            required=["intent", "reply"],
        )

    def parse_turn(self, ctx: TurnContext) -> TurnResult:
        from google.genai import types

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
            today=ctx.today.isoformat(),
            known_slots=json.dumps(ctx.known_slots),
            current_action=ctx.current_action,
        )
        contents = _build_contents(ctx)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_schema=self._response_schema(),
                    temperature=0.1,
                ),
            )
            raw = response.text
            usage = response.usage_metadata
            if usage is not None:
                self.total_prompt_tokens += usage.prompt_token_count or 0
                self.total_output_tokens += usage.candidates_token_count or 0
                self.total_calls += 1
        except Exception as exc:  # network error, quota, auth failure, etc.
            logger.error("Gemini call failed: %s", exc)
            raise IntentEngineError(str(exc)) from exc

        try:
            data = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            logger.error("Gemini returned non-JSON output: %r", raw)
            raise IntentEngineError(f"Could not parse model output: {exc}") from exc

        intent = data.get("intent", "unclear")
        if intent not in VALID_INTENTS:
            intent = "unclear"

        return TurnResult(
            reply=data.get("reply") or "Sorry, could you say that again?",
            intent=intent,
            name=data.get("name") or None,
            phone=data.get("phone") or None,
            date_iso=data.get("date_iso") or None,
            time_24h=data.get("time_24h") or None,
            duration_minutes=data.get("duration_minutes") or None,
            confirmed=data.get("confirmed"),
        )
