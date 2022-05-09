from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Dict

import datasets
import requests
from PIL import Image
from segments.utils import load_image_from_url, load_label_bitmap_from_url

# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from segments.typing import Release

logger = logging.getLogger(__name__)


def release2dataset(
    release: Release, download_images: bool = True
) -> datasets.Dataset:  # type:ignore
    """Create a Huggingface dataset from a release.

    Args:
        release: A Segments release.
        download_images: If images need to be downloaded from an AWS S3 url. Defaults to :obj:`True`.
    Returns:
        A HuggingFace dataset.
    Raises:
        ImportError: If HuggingFace datasets is not installed.
        ValueError: If the type of dataset is not yet supported.
    """
    try:
        import datasets
    except ImportError as e:
        logger.error(
            "Please install HuggingFace datasets first: pip install --upgrade datasets"
        )
        raise e

    content = requests.get(release.attributes.url)  # type:ignore
    release_dict = json.loads(content.content)

    task_type = release_dict["dataset"]["task_type"]

    if task_type in ["vector", "bboxes", "keypoint"]:
        features = datasets.Features(
            {
                "name": datasets.Value("string"),
                "image": {"url": datasets.Value("string")},
                "status": datasets.Value("string"),
                "label": {
                    "annotations": [
                        {
                            "id": datasets.Value("int32"),
                            "category_id": datasets.Value("int32"),
                            "type": datasets.Value("string"),
                            "points": [[datasets.Value("float32")]],
                        }
                    ]
                },
            }
        )

    elif task_type in ["segmentation-bitmap", "segmentation-bitmap-highres"]:
        features = datasets.Features(
            {
                "name": datasets.Value("string"),
                "image": {"url": datasets.Value("string")},
                "status": datasets.Value("string"),
                "label": {
                    "annotations": [
                        {
                            "id": datasets.Value("int32"),
                            "category_id": datasets.Value("int32"),
                        }
                    ],
                    "segmentation_bitmap": {"url": datasets.Value("string")},
                },
            }
        )

    elif task_type in ["text-named-entities", "text-span-categorization"]:
        features = datasets.Features(
            {
                "name": datasets.Value("string"),
                "text": datasets.Value("string"),
                "status": datasets.Value("string"),
                "label": {
                    "annotations": [
                        {
                            "start": datasets.Value("int32"),
                            "end": datasets.Value("int32"),
                            "category_id": datasets.Value("int32"),
                        }
                    ],
                },
            }
        )

    else:
        raise ValueError("This type of dataset is not yet supported.")

    samples = release_dict["dataset"]["samples"]

    data_rows = []
    for sample in samples:
        try:
            del sample["labels"]["ground-truth"]["attributes"]["format_version"]
        except (KeyError, TypeError):
            pass

        data_row: Dict[str, Any] = {}

        # Name
        data_row["name"] = sample["name"]

        # Status
        try:
            data_row["status"] = sample["labels"]["ground-truth"]["label_status"]
        except (KeyError, TypeError):
            data_row["status"] = "UNLABELED"

        # Image or text
        if task_type in [
            "vector",
            "bboxes",
            "keypoint",
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            try:
                data_row["image"] = sample["attributes"]["image"]
            except KeyError:
                data_row["image"] = {"url": None}
        elif task_type in ["text-named-entities", "text-span-categorization"]:
            try:
                data_row["text"] = sample["attributes"]["text"]
            except KeyError:
                data_row["text"] = None

        # Label
        try:
            data_row["label"] = sample["labels"]["ground-truth"]["attributes"]
        except (KeyError, TypeError):
            label: Dict[str, Any] = {"annotations": []}
            if task_type in ["segmentation-bitmap", "segmentation-bitmap-highres"]:
                label["segmentation_bitmap"] = {"url": None}
            data_row["label"] = label

        data_rows.append(data_row)

    logger.info(data_rows)

    # Now transform to column format
    dataset_dict: Dict[str, Any] = {key: [] for key in features.keys()}
    for data_row in data_rows:
        for key in dataset_dict.keys():
            dataset_dict[key].append(data_row[key])

    # Create the HF Dataset and flatten it
    dataset = datasets.Dataset.from_dict(dataset_dict, features)  # type:ignore
    dataset = dataset.flatten()  # type:ignore

    # Optionally download the images
    if (
        task_type
        in [
            "vector",
            "bboxes",
            "keypoint",
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]
        and download_images
    ):

        def download_image(data_row: Dict[str, Any]) -> Dict[str, Any]:
            try:
                data_row["image"] = load_image_from_url(data_row["image.url"])
            except Exception:
                data_row["image"] = None
            return data_row

        def download_segmentation_bitmap(data_row: Dict[str, Any]) -> Dict[str, Any]:
            try:
                segmentation_bitmap = load_label_bitmap_from_url(
                    data_row["label.segmentation_bitmap.url"]
                )
                data_row["label.segmentation_bitmap"] = Image.fromarray(
                    segmentation_bitmap
                )
            except Exception:
                data_row["label.segmentation_bitmap"] = Image.new(
                    "RGB", (1, 1)
                )  # None not possible?
            return data_row

        dataset = dataset.map(download_image, remove_columns=["image.url"])
        if task_type in ["segmentation-bitmap", "segmentation-bitmap-highres"]:
            dataset = dataset.map(
                download_segmentation_bitmap,
                remove_columns=["label.segmentation_bitmap.url"],
            )

    return dataset
