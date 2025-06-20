import pytest
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
        # Get the releases
        dataset_identifier = f"{owner}/{dataset.name}"
        releases = client.get_releases(dataset_identifier)

        for release in releases:
            # Get the dataset
            release = client.get_release(dataset_identifier, release.name)
            if dataset.task_type in SegmentsDataset.SUPPORTED_DATASET_TYPES:
                segments_dataset = SegmentsDataset(release, segments_dir=f"{ARTIFACTS_DIR}/segments")

                # Load the categories
                categories = segments_dataset.categories
                assert isinstance(categories, list)
                for category in categories:
                    assert isinstance(category, SegmentsDatasetCategory)

                # Iterate over samples
                for sample in segments_dataset:
                    assert isinstance(sample, dict)

            else:
                # Should throw an error for unsupported dataset types
                with pytest.raises(ValueError):
                    SegmentsDataset(release, segments_dir=f"{ARTIFACTS_DIR}/segments")


def test_dataset_no_image_load(client: SegmentsClient, owner: str):
    releases = client.get_releases("python-sdk-tests-organization/example-images-segmentation")
    release = releases[0]

    dataset = SegmentsDataset(release, preload=False, load_images=False)
    assert dataset[0]["image"] is None, "Image was loaded, but shouldn't have been"

    dataset = SegmentsDataset(release, preload=False, load_images=True)
    assert dataset[0]["image"] is not None, "Image should have been loaded, but not found!"
