"""TypeSafe client wired to project .env."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from typesafe_sdk import TypeSafeClient

ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    load_dotenv(ROOT / ".env")


def make_client(model: str = "jev-latest") -> TypeSafeClient:
    load_env()
    return TypeSafeClient(model=model)
