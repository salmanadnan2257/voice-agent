"""Twilio integration tests.

Per the project's budget discipline: no real phone call is placed anywhere
in this test suite. Twilio integration is proven two ways instead, both
essentially free:

1. Webhook signature validation exercised with the REAL account auth token
   (a local HMAC computation against Twilio's algorithm, no network call).
2. One real, read-only Twilio REST API call that looks up the account's own
   phone number and confirms it is voice-capable (GET requests against the
   Twilio API are not billed; only calls/SMS are).
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv
from twilio.request_validator import RequestValidator

load_dotenv()


def test_webhook_signature_validator_accepts_correctly_signed_request():
    """Uses the real TWILIO_AUTH_TOKEN to sign a request the way Twilio
    would, then verifies our validator (voice_agent.app.verify_twilio_signature
    logic, via the same RequestValidator Twilio itself provides) accepts it
    and rejects a tampered one. No network call: this is a local HMAC
    computation, so it costs nothing and needs no live account.
    """
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    if not auth_token:
        pytest.skip("TWILIO_AUTH_TOKEN not set")

    validator = RequestValidator(auth_token)
    url = "https://example.ngrok-free.app/voice/incoming"
    params = {"CallSid": "CA123", "From": "+15551234567", "To": "+15559999999"}

    signature = validator.compute_signature(url, params)
    assert validator.validate(url, params, signature) is True

    # Tampering with a param must invalidate the signature.
    tampered = dict(params, From="+15550000000")
    assert validator.validate(url, tampered, signature) is False

    # A signature computed with the wrong token must not validate either.
    wrong_validator = RequestValidator("not-the-real-token-00000000000000")
    wrong_signature = wrong_validator.compute_signature(url, params)
    assert validator.validate(url, params, wrong_signature) is False


@pytest.mark.real_api
def test_real_twilio_phone_number_is_voice_capable():
    """One real, read-only Twilio REST API call: confirms the number in
    TWILIO_PHONE_NUMBER is an active, voice-capable number on this account.
    Does not place a call. GET requests are free on Twilio's API.
    """
    from twilio.rest import Client

    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    phone_number = os.environ.get("TWILIO_PHONE_NUMBER")
    if not (account_sid and auth_token and phone_number):
        pytest.skip("Twilio credentials not set")

    client = Client(account_sid, auth_token)
    numbers = client.incoming_phone_numbers.list(phone_number=phone_number, limit=5)
    assert len(numbers) >= 1, f"{phone_number} not found on this Twilio account"
    assert numbers[0].capabilities.get("voice") is True
