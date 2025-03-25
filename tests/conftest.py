import json
import os
from typing import Iterable, List, Optional, cast

import pytest
from segments.client import SegmentsClient


pytest_plugins = ["tests.fixtures.payloads"]


@pytest.fixture(scope="module")
def API_KEY() -> str:
    return cast(str, os.getenv("SEGMENTS_API_KEY"))


@pytest.fixture(scope="module")
def API_URL() -> Optional[str]:
    return os.getenv("SEGMENTS_API_URL")


@pytest.fixture(scope="module")
def owner() -> str:
    return cast(str, os.getenv("DATASET_OWNER"))


@pytest.fixture(scope="module")
def client(request, API_KEY: str, API_URL: Optional[str]) -> Iterable[SegmentsClient]:
    client = SegmentsClient(api_key=API_KEY, api_url=API_URL) if API_URL else SegmentsClient(api_key=API_KEY)
    yield client
    client.close()


@pytest.fixture(scope="module")
def datasets() -> List[str]:
    datasets_str = cast(str, os.getenv("DATASETS"))
    datasets = cast(List[str], json.loads(datasets_str))
    return datasets


@pytest.fixture(scope="module")
def sample_uuids() -> List[str]:
    """First sample uuid for each dataset."""
    sample_uuids_str = cast(str, os.getenv("SAMPLE_UUIDS"))
    sample_uuids = cast(List[str], json.loads(sample_uuids_str))
    return sample_uuids


@pytest.fixture(scope="module")
def sample_attribute_types() -> List[str]:
    """Sample attribute type of the datasets."""
    sample_attribute_types_str = cast(str, os.getenv("SAMPLE_ATTRIBUTE_TYPES"))
    sample_attribute_types = cast(List[str], json.loads(sample_attribute_types_str))
    return sample_attribute_types


@pytest.fixture(scope="module")
def label_attribute_types() -> List[str]:
    """Label attribute type of the datasets."""
    label_attribute_types_str = cast(str, os.getenv("LABEL_ATTRIBUTE_TYPES"))
    label_attribute_types = cast(List[str], json.loads(label_attribute_types_str))
    return label_attribute_types


@pytest.fixture(scope="module")
def TIME_INTERVAL() -> float:
    """Wait for API call to complete."""
    return 0.2


@pytest.fixture(scope="module")
def ARTIFACTS_DIR() -> str:
    return "./test_artifacts"


@pytest.fixture(scope="class")
def setup_class_client(request, client):
    request.cls.client = client
