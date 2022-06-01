import json
import os
from typing import Iterable, List, Optional, cast

import pytest
from segments.client import SegmentsClient


@pytest.fixture
def API_KEY() -> str:
    return cast(str, os.getenv("SEGMENTS_API_KEY"))


@pytest.fixture
def API_URL() -> Optional[str]:
    return os.getenv("SEGMENTS_API_URL")


@pytest.fixture
def owner() -> str:
    return cast(str, os.getenv("DATASET_OWNER"))


@pytest.fixture
def client(API_KEY: str, API_URL: Optional[str]) -> Iterable[SegmentsClient]:
    client = (
        SegmentsClient(api_key=API_KEY, api_url=API_URL)
        if API_URL
        else SegmentsClient(api_key=API_KEY)
    )
    yield client
    client.close()


@pytest.fixture
def datasets() -> List[str]:
    datasets_str = cast(str, os.getenv("DATASETS"))
    datasets = cast(List[str], json.loads(datasets_str))
    return datasets


@pytest.fixture
def sample_uuids() -> List[str]:
    """First sample uuid for each dataset."""
    sample_uuids_str = cast(str, os.getenv("SAMPLE_UUIDS"))
    sample_uuids = cast(List[str], json.loads(sample_uuids_str))
    return sample_uuids


@pytest.fixture
def labelsets() -> List[str]:
    """Labelsets of first sample of first dataset."""
    labelsets_str = cast(str, os.getenv("LABELSETS"))
    labelsets = cast(List[str], json.loads(labelsets_str))
    return labelsets


@pytest.fixture
def releases() -> List[str]:
    """Releases of first dataset."""
    releases_str = cast(str, os.getenv("RELEASES"))
    releases = cast(List[str], json.loads(releases_str))
    return releases


@pytest.fixture
def sample_attribute_types() -> List[str]:
    """Sample attribute type of the datasets."""
    sample_attribute_types_str = cast(str, os.getenv("SAMPLE_ATTRIBUTE_TYPES"))
    sample_attribute_types = cast(List[str], json.loads(sample_attribute_types_str))
    return sample_attribute_types


@pytest.fixture
def label_attribute_types() -> List[str]:
    """Label attribute type of the datasets."""
    label_attribute_types_str = cast(str, os.getenv("LABEL_ATTRIBUTE_TYPES"))
    label_attribute_types = cast(List[str], json.loads(label_attribute_types_str))
    return label_attribute_types


@pytest.fixture
def TIME_INTERVAL() -> float:
    """Wait for API call to complete."""
    return 0.2


@pytest.fixture
def TMP_DIR() -> str:
    return "./tmp"
