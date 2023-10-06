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


##################
# Authentication #
##################
class AuthenticationError(SegmentsError):
    """Raised when an API key fails authentication."""

    pass


#################
# Authorization #
#################
class AuthorizationError(SegmentsError):
    """Raised when a user is unauthorized to perform the given request."""

    pass


########################
# Network (HTTP error) #
########################
class NetworkError(SegmentsError):
    """Raised when an HTTP error occurs."""

    pass


class NotFoundError(NetworkError):
    """Raised when the requested object is not found (e.g., because the name is misspelled, the page index is incorrect, ...)."""

    pass


class AlreadyExistsError(NetworkError):
    """Raised when the object (e.g., dataset, labelset, release, ...) already exists."""

    pass


class CollaboratorError(NetworkError):
    """Raised when a collaborator error occurs (e.g., a collaborator already has a role / cannot be added to an organization / ...)."""

    pass


class SubscriptionError(NetworkError):
    """Raised when a subscription error occurs (e.g., your trial ends, you exceed your user limit for private datasets, ...)."""

    pass


############
# Time out #
############
class TimeoutError(SegmentsError):
    """Raised when a request times out."""

    pass


#############
# API limit #
#############
class APILimitError(SegmentsError):
    """Raised when the user performs too many requests in a period of time."""

    pass


##############
# Validation #
##############
class ValidationError(SegmentsError):
    """Raised when validation of the response fails."""

    pass
