"""Outbound test-call CLI.

Places a REAL Twilio phone call to a number the owner passes explicitly as a
command-line argument, wiring it to this service's /voice/incoming TwiML
endpoint. Never called automatically by anything else in this repo: it is
only invoked by a human running it from a terminal.

Usage:
    python -m voice_agent.cli +15551234567 --base-url https://your-ngrok-id.ngrok-free.app

Requires a public URL Twilio can reach (ngrok tunnel or a real deploy),
either via --base-url or the PUBLIC_WEBHOOK_BASE_URL env var, because Twilio
fetches TwiML instructions from that URL over the public internet once the
call connects. Refuses to run without one rather than placing a call that
Twilio cannot actually complete meaningfully.
"""
from __future__ import annotations

import argparse
import re
import sys

from dotenv import load_dotenv

from . import config

E164_RE = re.compile(r"^\+[1-9]\d{6,14}$")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Place a real outbound Twilio test call to a number you control."
    )
    parser.add_argument("phone_number", help="E.164 destination number, e.g. +15551234567")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Public base URL for this service (overrides PUBLIC_WEBHOOK_BASE_URL env var)",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt before placing the real call",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if not E164_RE.match(args.phone_number):
        print(f"Error: {args.phone_number!r} does not look like an E.164 number (e.g. +15551234567)")
        return 1

    base_url = args.base_url or config.public_webhook_base_url()
    if not base_url:
        print(
            "Error: no public webhook URL configured. This CLI cannot place a usable test "
            "call without one, because Twilio needs to fetch TwiML instructions from a URL "
            "it can reach over the internet.\n"
            "Set one up (e.g. `ngrok http 8000`, or a real deploy) and either pass "
            "--base-url https://... or set PUBLIC_WEBHOOK_BASE_URL in .env, then retry."
        )
        return 1

    try:
        twilio_cfg = config.load_twilio_config()
    except config.ConfigError as exc:
        print(f"Error: {exc}")
        return 1

    if not args.yes:
        answer = input(
            f"About to place a REAL outbound call from {twilio_cfg.phone_number} to "
            f"{args.phone_number}. This will incur real Twilio charges. Continue? [y/N] "
        )
        if answer.strip().lower() not in ("y", "yes"):
            print("Aborted, no call placed.")
            return 1

    from twilio.rest import Client

    client = Client(twilio_cfg.account_sid, twilio_cfg.auth_token)
    voice_url = base_url.rstrip("/") + "/voice/incoming"
    status_url = base_url.rstrip("/") + "/voice/status"
    call = client.calls.create(
        to=args.phone_number,
        from_=twilio_cfg.phone_number,
        url=voice_url,
        status_callback=status_url,
        status_callback_event=["completed"],
    )
    print(f"Call placed. SID={call.sid} status={call.status} to={args.phone_number}")
    print(f"TwiML source: {voice_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
