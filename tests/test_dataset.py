from typing import List, cast

from segments.client import SegmentsClient
from segments.dataset import SegmentsDataset
from segments.utils import export_dataset
from typing_extensions import Final
from segments.typing import SegmentsDatasetCategory


def test_dataset(
    client: SegmentsClient,
    owner: str,
    datasets: List[str],
    releases: List[str],
) -> None:

    # Get the dataset
    dataset_identifier, name = f"{owner}/{datasets[0]}", releases[0]
    release = client.get_release(dataset_identifier, name)
    dataset = SegmentsDataset(release)

    # Load the categories
    categories = dataset.categories
    assert isinstance(categories, list)
    for category in categories:
        assert isinstance(category, SegmentsDatasetCategory)

    # Iterate over samples
    for sample in dataset:
        assert isinstance(sample, dict)


def test_export_dataset(
    client: SegmentsClient,
    owner: str,
    datasets: List[str],
    releases: List[str],
) -> None:

    # Get the dataset
    dataset_identifier, name = f"{owner}/{datasets[0]}", releases[0]
    release = client.get_release(dataset_identifier, name)
    dataset = SegmentsDataset(release)

    # Export the dataset
    export_formats = [
        "coco-panoptic",
        "coco-instance",
        "yolo",
        "instance",
        "instance-color",
        "semantic",
        "semantic-color",
    ]
    export_folder = "./tmp"
    for export_format in export_formats:
        export_dataset(dataset, export_folder, export_format)  # type:ignore
