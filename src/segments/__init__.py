import logging

from segments.client import SegmentsClient
from segments.dataset import SegmentsDataset

# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

__all__ = [
    "SegmentsClient",
    "SegmentsDataset",
]
