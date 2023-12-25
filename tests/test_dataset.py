from segments.client import SegmentsClient
from segments.dataset import SegmentsDataset
from segments.typing import SegmentsDatasetCategory


def test_dataset(
    client: SegmentsClient,
    owner: str,
    ARTIFACTS_DIR: str,
) -> None:
    # Get the datasets
    datasets = client.get_datasets(owner)

    for dataset in datasets:
        # Skip the example-multi-sensor dataset
        if dataset.name == "example-multi-sensor" or dataset.name == "example-point-cloud-sequences-segmentation":
            continue

        # Get the releases
        dataset_identifier = f"{owner}/{dataset.name}"
        releases = client.get_releases(dataset_identifier)

        for release in releases:
            # Get the dataset
            release = client.get_release(dataset_identifier, release.name)
            segments_dataset = SegmentsDataset(release, segments_dir=f"{ARTIFACTS_DIR}/segments")

            # Load the categories
            categories = segments_dataset.categories
            assert isinstance(categories, list)
            for category in categories:
                assert isinstance(category, SegmentsDatasetCategory)

            # Iterate over samples
            for sample in segments_dataset:
                assert isinstance(sample, dict)
