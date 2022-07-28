from typing import List

import pytest
from segments.client import SegmentsClient
from segments.dataset import SegmentsDataset
from segments.utils import export_dataset


def test_export_dataset(
    client: SegmentsClient,
    owner: str,
    datasets: List[str],
    releases: List[str],
    ARTIFACTS_DIR: str,
) -> None:

    # Get the dataset
    dataset_identifier, name = f"{owner}/{datasets[0]}", releases[0]
    release = client.get_release(dataset_identifier, name)
    dataset = SegmentsDataset(release, segments_dir=f"{ARTIFACTS_DIR}/segments")

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
    export_folder = f"{ARTIFACTS_DIR}"
    for export_format in export_formats:
        if export_format == "yolo":
            with pytest.raises(ValueError):
                export_dataset(dataset, export_folder, export_format)
        else:
            export_dataset(dataset, export_folder, export_format)
