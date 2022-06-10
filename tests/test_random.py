from segments import SegmentsClient
from segments.exceptions import AuthenticationError
import pytest


def test_random() -> None:
    API_KEY = "0"
    with pytest.raises(AuthenticationError):
        client = SegmentsClient(API_KEY)
