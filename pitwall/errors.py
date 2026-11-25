"""Exceptions raised by pitwall."""


class PitwallError(Exception):
    """Base class for every pitwall error."""


class DataNotAvailableError(PitwallError):
    """The data does not exist (race not run yet, too old, or source lacks it)."""


class RateLimitError(PitwallError):
    """Another request would break the data provider's rate limit."""


class SourceRequestError(PitwallError):
    """A request to a data source failed, even after retrying."""


class AuthenticationError(PitwallError):
    """Your own credentials are missing, expired or were rejected."""


class AccessBlockedError(PitwallError):
    """The provider refused the request outright. pitwall never works around this."""
