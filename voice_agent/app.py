"""FastAPI service Twilio calls for TwiML instructions.

Endpoints:
  GET  /health          liveness check
  POST /voice/incoming  answers a call, greets, opens the first <Gather>
  POST /voice/gather    receives transcribed speech, runs the agent, replies
  POST /voice/status    Twilio call status callback, used to detect abandoned calls

Run locally with: uvicorn voice_agent.app:app --reload
No public URL is required to boot this, hit /health, or exercise the
endpoints directly (see tests/test_twiml_endpoints.py). A public URL
(PUBLIC_WEBHOOK_BASE_URL, e.g. an ngrok tunnel) is only needed for a real
Twilio phone number to actually reach this service over the internet.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from twilio.request_validator import RequestValidator

from . import config
from .calendar_store import CalendarStore, get_calendar_store
from .conversation import ConversationManager
from .intent_engine import GeminiIntentEngine, IntentEngine
from .session_store import SessionStore
from .twiml import build_gather_twiml, build_hangup_twiml

load_dotenv()
logger = logging.getLogger("voice_agent.app")

GREETING = (
    "Thanks for calling. I can help you book, reschedule, or cancel an appointment. "
    "What would you like to do?"
)


def verify_twilio_signature(request: Request, form: dict, auth_token: str, base_url: str) -> bool:
    """Validates Twilio's X-Twilio-Signature header against the exact URL and
    form body. Only meaningful once PUBLIC_WEBHOOK_BASE_URL is configured,
    since Twilio signs the public URL it actually requested, and we need
    that same URL string to recompute the signature.
    """
    validator = RequestValidator(auth_token)
    signature = request.headers.get("X-Twilio-Signature", "")
    url = base_url.rstrip("/") + request.url.path
    return validator.validate(url, form, signature)


def create_app(
    calendar_store: Optional[CalendarStore] = None,
    session_store: Optional[SessionStore] = None,
    intent_engine: Optional[IntentEngine] = None,
    enforce_signature: bool = True,
) -> FastAPI:
    """App factory. Passing explicit stores/engine (as the tests do) avoids
    touching real Gemini or a shared database file; passing nothing (as
    __main__ / uvicorn does) wires up the real production dependencies.
    """
    app = FastAPI(title="voice-agent")

    app.state.calendar_store = calendar_store or get_calendar_store()
    app.state.session_store = session_store or SessionStore(config.db_path())
    if intent_engine is not None:
        app.state.intent_engine = intent_engine
    else:
        gemini_cfg = config.load_gemini_config()
        app.state.intent_engine = GeminiIntentEngine(
            project=gemini_cfg.project, location=gemini_cfg.location, model=gemini_cfg.model
        )
    app.state.conversation_manager = ConversationManager(
        intent_engine=app.state.intent_engine,
        calendar_store=app.state.calendar_store,
        session_store=app.state.session_store,
    )
    app.state.enforce_signature = enforce_signature

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/voice/incoming")
    async def voice_incoming(request: Request):
        form = dict(await request.form())
        call_sid = form.get("CallSid", "unknown-call")
        caller = form.get("From", "unknown")

        if app.state.enforce_signature:
            base_url = config.public_webhook_base_url()
            auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
            if base_url and auth_token:
                if not verify_twilio_signature(request, form, auth_token, base_url):
                    return Response(status_code=403, content="Invalid Twilio signature")

        session_store: SessionStore = app.state.session_store
        session = session_store.get_or_create(call_sid, caller)
        session_store.log_turn(call_sid, 0, "agent", GREETING)

        gather_url = _gather_url(request)
        twiml = build_gather_twiml(GREETING, gather_url)
        return Response(content=twiml, media_type="application/xml")

    @app.post("/voice/gather")
    async def voice_gather(request: Request):
        form = dict(await request.form())
        call_sid = form.get("CallSid", "unknown-call")
        caller = form.get("From", "unknown")
        speech = (form.get("SpeechResult") or "").strip()

        if app.state.enforce_signature:
            base_url = config.public_webhook_base_url()
            auth_token = os.environ.get("TWILIO_AUTH_TOKEN", "")
            if base_url and auth_token:
                if not verify_twilio_signature(request, form, auth_token, base_url):
                    return Response(status_code=403, content="Invalid Twilio signature")

        session_store: SessionStore = app.state.session_store
        conversation_manager: ConversationManager = app.state.conversation_manager

        session = session_store.get_or_create(call_sid, caller)
        session_store.log_turn(call_sid, session.turn_count, "caller", speech or "[no speech detected]")

        turn_outcome = conversation_manager.handle_turn(session, speech)

        session_store.log_turn(call_sid, session.turn_count, "agent", turn_outcome.reply)

        if turn_outcome.end_call:
            twiml = build_hangup_twiml(turn_outcome.reply)
        else:
            gather_url = _gather_url(request)
            twiml = build_gather_twiml(turn_outcome.reply, gather_url)
        return Response(content=twiml, media_type="application/xml")

    @app.post("/voice/status")
    async def voice_status(request: Request):
        form = dict(await request.form())
        call_sid = form.get("CallSid", "unknown-call")
        call_status = form.get("CallStatus", "")
        if call_status in ("completed", "failed", "busy", "no-answer", "canceled"):
            app.state.session_store.mark_abandoned_if_unresolved(call_sid)
        return Response(status_code=204)

    return app


def _gather_url(request: Request) -> str:
    base_url = config.public_webhook_base_url()
    if base_url:
        return base_url.rstrip("/") + "/voice/gather"
    # Falls back to whatever host this request actually arrived on (fine for
    # local testing with TestClient; a real Twilio number needs the public
    # base URL set so Twilio, calling in from the internet, gets a URL it
    # can reach back).
    return str(request.url_for("voice_gather"))


# Real, production app instance (used by `uvicorn voice_agent.app:app`).
# Building it touches the real Gemini client and the real SQLite file, which
# needs valid credentials in the environment. Constructed lazily via module
# __getattr__ (PEP 562) so merely importing this module (as every test does,
# via `create_app`) never requires those credentials to be present; only
# actually accessing the `app` attribute (what uvicorn does) builds it.
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
