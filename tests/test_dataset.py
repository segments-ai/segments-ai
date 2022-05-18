from typing import List

from segments.client import SegmentsClient
from segments.dataset import SegmentsDataset
from segments.typing import SegmentsDatasetCategory


def test_dataset(
    client: SegmentsClient,
    owner: str,
    datasets: List[str],
    releases: List[str],
    TMP_DIR: str,
) -> None:

    # Get the dataset
    dataset_identifier, name = f"{owner}/{datasets[0]}", releases[0]
    release = client.get_release(dataset_identifier, name)
    dataset = SegmentsDataset(release, segments_dir=f"{TMP_DIR}/segments")

    # Load the categories
    categories = dataset.categories
    assert isinstance(categories, list)
    for category in categories:
        assert isinstance(category, SegmentsDatasetCategory)

    # Iterate over samples
    for sample in dataset:
        assert isinstance(sample, dict)
