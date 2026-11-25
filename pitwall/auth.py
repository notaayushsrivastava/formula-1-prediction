"""Your own F1 TV token, read from an environment variable.

pitwall never stores, caches, prints or logs the token.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
import os

from .errors import AuthenticationError

TOKEN_ENV_VAR = "PITWALL_F1TV_TOKEN"
HOW_TO = (
    f"Set {TOKEN_ENV_VAR} to the subscription token of YOUR OWN F1 TV Access (or higher) "
    f"account: `export {TOKEN_ENV_VAR}=...` (Windows: `set {TOKEN_ENV_VAR}=...`). "
    "Tokens expire after a few days; sign in again to get a fresh one."
)


def get_token() -> str:
    """The token without any 'Bearer ' prefix. Raises if it is missing or expired."""
    token = os.environ.get(TOKEN_ENV_VAR, "").strip().removeprefix("Bearer ").strip()
    if not token:
        raise AuthenticationError(f"No F1 TV token found. {HOW_TO}")
    expires = token_expiry(token)
    if expires is not None and expires <= dt.datetime.now(dt.timezone.utc):
        raise AuthenticationError(
            f"Your F1 TV token expired at {expires:%Y-%m-%d %H:%M} UTC. {HOW_TO}")
    return token


def auth_header(token: str) -> dict[str, str]:
    """The HTTP header F1's archive expects."""
    return {"Authorization": f"Bearer {token}"}


def token_expiry(token: str) -> dt.datetime | None:
    """Expiry time if the token is a JWT with an 'exp' claim, otherwise None."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)  # base64 needs length % 4 == 0
        payload = json.loads(base64.urlsafe_b64decode(padded))
        return dt.datetime.fromtimestamp(float(payload["exp"]), tz=dt.timezone.utc)
    except (ValueError, KeyError, TypeError):
        return None


def token_status() -> str:
    """Whether a usable token is set, WITHOUT revealing it."""
    try:
        token = get_token()
    except AuthenticationError as exc:
        return str(exc)
    expires = token_expiry(token)
    when = f"expires {expires:%Y-%m-%d %H:%M} UTC" if expires else "expiry unknown"
    return f"F1 TV token found ({len(token)} characters, {when})."
