import pytest
from segments import SegmentsClient
from segments.exceptions import AuthenticationError


def test_random() -> None:
    API_KEY = "0"
    with pytest.raises(AuthenticationError):
        SegmentsClient(API_KEY)
