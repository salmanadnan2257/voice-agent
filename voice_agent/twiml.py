"""TwiML generation. Kept separate from app.py so it can be unit tested (and
read) without spinning up FastAPI at all.
"""
from __future__ import annotations

from twilio.twiml.voice_response import Gather, VoiceResponse

GATHER_TIMEOUT_SECONDS = 6


def build_gather_twiml(say_text: str, gather_action_url: str) -> str:
    """Say something, then listen for speech. Every outcome (speech captured,
    silence, or garbled audio) posts back to gather_action_url: silence and
    garbled audio are routed there too via actionOnEmptyResult, so the
    conversation manager's own retry/handoff logic handles them, rather than
    needing separate fallback TwiML branches here.
    """
    response = VoiceResponse()
    gather = Gather(
        input="speech",
        action=gather_action_url,
        method="POST",
        speech_timeout="auto",
        timeout=GATHER_TIMEOUT_SECONDS,
        action_on_empty_result=True,
    )
    gather.say(say_text)
    response.append(gather)
    # Belt and suspenders: only reached if Twilio somehow does not re-request
    # the action URL (e.g. the call already ended).
    response.say("We didn't receive any input. Goodbye.")
    response.hangup()
    return str(response)


def build_hangup_twiml(say_text: str) -> str:
    response = VoiceResponse()
    response.say(say_text)
    response.hangup()
    return str(response)
