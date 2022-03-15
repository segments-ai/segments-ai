import json
from typing import Any, Dict, Optional
import requests
from PIL import Image

from typehints import Release
from .utils import load_image_from_url, load_label_bitmap_from_url
import datasets


def release2dataset(
    release: Release, download_images: bool = True
) -> Optional[datasets.Dataset]:
    try:
        import datasets
    except ImportError:
        print(
            "Please install HuggingFace datasets first: pip install --upgrade datasets"
        )
        return None

    content = requests.get(release.attributes.url)
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
        print("This type of dataset is not yet supported.")
        assert False

    samples = release_dict["dataset"]["samples"]

    data_rows = []
    for sample in samples:
        try:
            del sample["labels"]["ground-truth"]["attributes"]["format_version"]
        except:
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

    print(data_rows)

    # Now transform to column format
    dataset_dict: Dict[str, Any] = {key: [] for key in features.keys()}
    for data_row in data_rows:
        for key in dataset_dict.keys():
            dataset_dict[key].append(data_row[key])

    # Create the HF Dataset and flatten it
    dataset = datasets.Dataset.from_dict(dataset_dict, features)
    dataset = dataset.flatten()

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
            except:
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
            except:
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
