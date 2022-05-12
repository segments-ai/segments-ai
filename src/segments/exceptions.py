from typing import Optional


class SegmentsError(Exception):
    """Base class for exceptions."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        """
        Args:
            message: An informative message about the exception.
            cause: The cause of the exception raised by Python or another library. Defaults to :obj:`None`.
        """

        super().__init__(message, cause)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        return self.message


class AuthenticationError(SegmentsError):
    """Raised when an API key fails authentication."""

    pass


class AuthorizationError(SegmentsError):
    """Raised when a user is unauthorized to perform the given request."""

    pass


class NetworkError(SegmentsError):
    """Raised when an HTTPError occurs."""


class TimeoutError(SegmentsError):
    """Raised when a request times out."""

    pass


class APILimitError(SegmentsError):
    """Raised when the user performs too many requests in a period of time."""

    pass


class ValidationError(SegmentsError):
    """Raised when Pydantic validation of the response fails."""

    pass