"""Environment configuration.

Every value is read from ``os.environ`` at call time, never cached at import
time and never logged. This module does not read the .env file itself; call
``load_dotenv()`` once at process start (done in ``app.py`` and ``cli.py``)
before touching anything in here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"Required environment variable {name} is not set")
    return value


@dataclass(frozen=True)
class TwilioConfig:
    account_sid: str
    auth_token: str
    phone_number: str


@dataclass(frozen=True)
class GeminiConfig:
    project: str
    location: str
    model: str = "gemini-2.5-flash"


def load_twilio_config() -> TwilioConfig:
    return TwilioConfig(
        account_sid=_require("TWILIO_ACCOUNT_SID"),
        auth_token=_require("TWILIO_AUTH_TOKEN"),
        phone_number=_require("TWILIO_PHONE_NUMBER"),
    )


def load_gemini_config() -> GeminiConfig:
    return GeminiConfig(
        project=_require("GCP_PROJECT"),
        location=os.environ.get("GCP_LOCATION", "global").strip() or "global",
    )


def public_webhook_base_url() -> str | None:
    """Base URL Twilio should hit for TwiML (e.g. an ngrok URL). None if unset."""
    value = os.environ.get("PUBLIC_WEBHOOK_BASE_URL", "").strip()
    return value or None


def google_calendar_credentials_path() -> str | None:
    value = os.environ.get("GOOGLE_CALENDAR_CREDENTIALS_PATH", "").strip()
    return value or None


def db_path() -> str:
    """Path to the SQLite database file. Overridable for tests via env var."""
    return os.environ.get("VOICE_AGENT_DB_PATH", "").strip() or _default_db_path()


def _default_db_path() -> str:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(here, "data")
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, "voice_agent.db")


# Business rules, kept simple and centralized. A single-location agent with
# no timezone conversion: all times are the business's local wall-clock time.
BUSINESS_HOURS_START = 9  # 9:00 local
BUSINESS_HOURS_END = 17  # 17:00 local, last bookable start is 16:xx for a 30 min slot
DEFAULT_APPOINTMENT_DURATION_MINUTES = 30
MAX_RETRY_COUNT = 3  # consecutive unclear turns before handoff
MAX_TURN_COUNT = 14  # total turns before forced handoff (avoid infinite loops)
