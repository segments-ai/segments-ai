from __future__ import annotations
import json
import os
from typing import Iterable

from pytest import fixture
from segments.client import SegmentsClient


@fixture
def API_KEY() -> str:
    api_key: str = os.getenv("SEGMENTS_API_KEY")
    return api_key


@fixture
def API_URL() -> str | None:
    api_url: str | None = os.getenv("SEGMENTS_API_URL")
    return api_url


@fixture
def owner() -> str:
    dataset_owner: str = os.getenv("DATASET_OWNER")
    return dataset_owner


@fixture
def client(API_KEY: str, API_URL: str | None) -> Iterable[SegmentsClient]:
    client = (
        SegmentsClient(api_key=API_KEY, api_url=API_URL)
        if API_URL
        else SegmentsClient(api_key=API_KEY)
    )
    yield client
    client.close()


@fixture
def datasets() -> list[str]:
    datasets: str = os.getenv("DATASETS")
    datasets: list[str] = json.loads(datasets)
    return datasets


@fixture
def sample_uuids() -> list[str]:
    """First sample uuid for each dataset."""
    sample_uuids: str = os.getenv("SAMPLE_UUIDS")
    sample_uuids: list[str] = json.loads(sample_uuids)
    return sample_uuids


@fixture
def sample_attribute_types() -> list[str]:
    """Sample attribute type of the datasets."""
    sample_attribute_types: str = os.getenv("SAMPLE_ATTRIBUTE_TYPES")
    sample_attribute_types: list[str] = json.loads(sample_attribute_types)
    return sample_attribute_types


@fixture
def label_attribute_types() -> list[str]:
    """Label attribute type of the datasets."""
    label_attribute_types: str = os.getenv("LABEL_ATTRIBUTE_TYPES")
    label_attribute_types: list[str] = json.loads(label_attribute_types)
    return label_attribute_types


@fixture
def TIME_INTERVAL() -> float:
    """Wait for API call to complete."""
    return 0.2


@fixture
def ARTIFACTS_DIR() -> str:
    return "./test_artifacts"
