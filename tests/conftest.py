from __future__ import annotations
import json
import os
from typing import Iterable, cast

import pytest
from segments.client import SegmentsClient


@pytest.fixture
def API_KEY() -> str:
    return cast(str, os.getenv("SEGMENTS_API_KEY"))


@pytest.fixture
def API_URL() -> str | None:
    return os.getenv("SEGMENTS_API_URL")


@pytest.fixture
def owner() -> str:
    return cast(str, os.getenv("DATASET_OWNER"))


@pytest.fixture
def client(API_KEY: str, API_URL: str | None) -> Iterable[SegmentsClient]:
    client = (
        SegmentsClient(api_key=API_KEY, api_url=API_URL)
        if API_URL
        else SegmentsClient(api_key=API_KEY)
    )
    yield client
    client.close()


@pytest.fixture
def datasets() -> list[str]:
    datasets_str = cast(str, os.getenv("DATASETS"))
    datasets = cast(list[str], json.loads(datasets_str))
    return datasets


@pytest.fixture
def sample_uuids() -> list[str]:
    """First sample uuid for each dataset."""
    sample_uuids_str = cast(str, os.getenv("SAMPLE_UUIDS"))
    sample_uuids = cast(list[str], json.loads(sample_uuids_str))
    return sample_uuids


@pytest.fixture
def sample_attribute_types() -> list[str]:
    """Sample attribute type of the datasets."""
    sample_attribute_types_str = cast(str, os.getenv("SAMPLE_ATTRIBUTE_TYPES"))
    sample_attribute_types = cast(list[str], json.loads(sample_attribute_types_str))
    return sample_attribute_types


@pytest.fixture
def label_attribute_types() -> list[str]:
    """Label attribute type of the datasets."""
    label_attribute_types_str = cast(str, os.getenv("LABEL_ATTRIBUTE_TYPES"))
    label_attribute_types = cast(list[str], json.loads(label_attribute_types_str))
    return label_attribute_types


@pytest.fixture
def TIME_INTERVAL() -> float:
    """Wait for API call to complete."""
    return 0.2


@pytest.fixture
def ARTIFACTS_DIR() -> str:
    return "./test_artifacts"
