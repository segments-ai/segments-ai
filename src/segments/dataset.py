from __future__ import annotations

import json
import logging
import os
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
import requests
from PIL import Image, UnidentifiedImageError
from pydantic import TypeAdapter
from segments.typing import LabelStatus, Release, SegmentsDatasetCategory
from segments.utils import (
    handle_exif_rotation,
    load_image_from_url,
    load_label_bitmap_from_url,
)
from tqdm import tqdm


#############
# Variables #
#############
logger = logging.getLogger(__name__)


class SegmentsDataset:
    """A class that represents a Segments dataset.

    .. code-block:: python

        # pip install --upgrade segments-ai
        from segments import SegmentsClient, SegmentsDataset
        from segments.utils import export_dataset

        # Initialize a SegmentsDataset from the release file
        client = SegmentsClient('YOUR_API_KEY')
        release = client.get_release('jane/flowers', 'v1.0') # Alternatively: release = 'flowers-v1.0.json'
        dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['LABELED', 'REVIEWED'])

        # Export to COCO panoptic format
        export_format = 'coco-panoptic'
        export_dataset(dataset, export_format)

    Alternatively, you can use the initialized :class:`SegmentsDataset` to loop through the samples and labels, and visualize or process them in any way you please:

    .. code-block:: python

        import matplotlib.pyplot as plt
        from segments.utils import get_semantic_bitmap

        for sample in dataset:

            # Print the sample name and list of labeled objects
            print(sample['name'])
            print(sample['annotations'])

            # Show the image
            plt.imshow(sample['image'])
            plt.show()

            # Show the instance segmentation label
            plt.imshow(sample['segmentation_bitmap'])
            plt.show()

            # Show the semantic segmentation label
            semantic_bitmap = get_semantic_bitmap(sample['segmentation_bitmap'], sample['annotations'])
            plt.imshow(semantic_bitmap)
            plt.show()

    Args:
        release_file: Path to a release file, or a release class resulting from :meth:`.get_release`.
        labelset: The labelset that should be loaded. Defaults to ``ground-truth``.
        filter_by: A list of label statuses to filter by. Defaults to :obj:`None`.
        filter_by_metadata: A dict of metadata key:value pairs to filter by. Filters are ANDed together. Defaults to :obj:`None`.
        segments_dir: The directory where the data will be downloaded to for caching. Set to :obj:`None` to disable caching. Defaults to ``segments``. Alternatively, you can set the ``SEGMENTS_DIR`` environment variable to change the default.
        preload: Whether the data should be pre-downloaded when the dataset is initialized. Ignored if ``segments_dir`` is :obj:`None`. Defaults to :obj:`True`.
        s3_client: A boto3 S3 client, e.g. ``s3_client = boto3.client("s3")``. Needs to be provided if your images are in a private S3 bucket. Defaults to :obj:`None`.

    Raises:
        :exc:`ValueError`: If the release task type is not one of: ``segmentation-bitmap``, ``segmentation-bitmap-highres``, ``image-vector-sequence``, ``bboxes``, ``vector``, ``pointcloud-cuboid``, ``pointcloud-cuboid-sequence``, ``pointcloud-segmentation``, ``pointcloud-segmentation-sequence``, ``text-named-entities``, or ``text-span-categorization``.
        :exc:`ValueError`: If there is no labelset with this name.
    """

    # https://stackoverflow.com/questions/682504/what-is-a-clean-pythonic-way-to-have-multiple-constructors-in-python
    def __init__(
        self,
        release_file: Union[str, Release],
        labelset: str = "ground-truth",
        filter_by: Optional[Union[LabelStatus, List[LabelStatus]]] = None,
        filter_by_metadata: Optional[Dict[str, str]] = None,
        segments_dir: str = "segments",
        preload: bool = True,
        s3_client: Optional[Any] = None,
    ):
        # check environment for SEGMENTS_DIR variable if `segments_dir` has default value
        if segments_dir == "segments":
            segments_dir = os.getenv("SEGMENTS_DIR", "segments")

        self.labelset = labelset
        if isinstance(filter_by, list):
            filter_by = [f.upper() for f in filter_by]
        elif filter_by:
            filter_by = [filter_by.upper()]
        self.filter_by = filter_by
        # if self.filter_by:
        #     self.filter_by = [s.lower() for s in self.filter_by]
        self.filter_by_metadata = filter_by_metadata
        self.segments_dir = segments_dir
        self.caching_enabled = segments_dir is not None
        self.preload = preload
        self.s3_client = s3_client

        # if urlparse(release_file).scheme in ('http', 'https'): # If it's a url
        if isinstance(release_file, str):  # If it's a file path
            with open(release_file) as f:
                self.release = json.load(f)
        else:  # If it's a release object
            release_file_url = release_file.attributes.url
            content = requests.get(cast(str, release_file_url))  # TODO Fix in backend.
            self.release = json.loads(content.content)
        self.release_file = release_file

        self.dataset_identifier = f"{self.release['dataset']['owner']}_{self.release['dataset']['name']}"

        self.image_dir = (
            None if segments_dir is None else os.path.join(segments_dir, self.dataset_identifier, self.release["name"])
        )

        # First some checks
        if self.labelset not in [labelset["name"] for labelset in self.release["dataset"]["labelsets"]]:
            raise ValueError(f"There is no labelset with name '{self.labelset}'.")

        self.task_type = self.release["dataset"]["task_type"]
        if self.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
            "vector",
            "bboxes",
            "keypoints",
            "image-vector-sequence",
            "pointcloud-cuboid",
            "pointcloud-segmentation",
        ]:
            raise ValueError(
                f"You can only create a dataset for tasks of type 'segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes', 'keypoints', 'image-vector-sequence', 'pointcloud-cuboid', 'pointcloud-segmentation' for now. Got {self.task_type}."
            )

        self.load_dataset()

    def load_dataset(self) -> None:
        print("Initializing dataset...")

        # Setup cache
        if self.caching_enabled and self.image_dir and not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        # Load and filter the samples
        samples = self.release["dataset"]["samples"]
        if self.filter_by:
            filtered_samples = []
            for sample in samples:
                if sample["labels"][self.labelset]:
                    label_status = sample["labels"][self.labelset]["label_status"]
                else:
                    label_status = "UNLABELED"

                if self.filter_by and label_status in self.filter_by:
                    filtered_samples.append(sample)
            samples = filtered_samples

        if self.filter_by_metadata:
            filtered_samples = []
            for sample in samples:
                # https://stackoverflow.com/a/41579450/1542912
                if self.filter_by_metadata.items() <= sample["metadata"].items():
                    filtered_samples.append(sample)
            samples = filtered_samples

        self.samples = samples

        # # Preload all samples (sequentially)
        # for i in tqdm(range(self.__len__())):
        #     item = self.__getitem__(i)

        # To avoid memory overflow or "Too many open files" error when using tqdm in combination with multiprocessing.
        def _load_image(i: int) -> int:
            self.__getitem__(i)
            return i

        # Preload all samples (in parallel)
        # https://stackoverflow.com/questions/16181121/a-very-simple-multithreading-parallel-url-fetching-without-queue/27986480
        # https://stackoverflow.com/questions/3530955/retrieve-multiple-urls-at-once-in-parallel
        # https://github.com/tqdm/tqdm/issues/484#issuecomment-461998250
        num_samples = self.__len__()
        if (
            self.caching_enabled
            and self.preload
            and self.task_type not in ["pointcloud-segmentation", "pointcloud-cuboid"]
        ):
            print("Preloading all samples. This may take a while...")
            with ThreadPool(16) as pool:
                # list(tqdm(pool.imap_unordered(self.__getitem__, range(num_samples)), total=num_samples))
                list(
                    tqdm(
                        pool.imap_unordered(_load_image, range(num_samples)),
                        total=num_samples,
                        colour="#FF9900",
                    )
                )

        print(f"Initialized dataset with {num_samples} images.")

    def _load_image_from_cache(self, sample: Dict[str, Any]) -> Tuple[Optional[Image.Image], str]:
        sample_name = os.path.splitext(sample["name"])[0]
        image_url = sample["attributes"]["image"]["url"]
        image_url_parsed = urlparse(image_url)
        url_extension = os.path.splitext(image_url_parsed.path)[1]
        # image_filename_rel = '{}{}'.format(sample['uuid'], url_extension)
        image_filename_rel = f"{sample_name}{url_extension}"

        if image_url_parsed.scheme == "s3":
            image = None
        else:
            if self.caching_enabled:
                image_filename = os.path.join(self.image_dir, image_filename_rel)
                if not os.path.exists(image_filename):
                    image = load_image_from_url(image_url, image_filename, self.s3_client)
                else:
                    try:
                        image = Image.open(image_filename)
                    except UnidentifiedImageError:
                        image = None
                        logger.error(f"Something went wrong loading image: {image_filename}")
            else:
                image = load_image_from_url(image_url, self.s3_client)

            image = handle_exif_rotation(image)

        return image, image_filename_rel

    def _load_segmentation_bitmap_from_cache(
        self, sample: Dict[str, Any], labelset: str
    ) -> Union[npt.NDArray[np.uint32], Image.Image]:
        sample_name = os.path.splitext(sample["name"])[0]
        label = sample["labels"][labelset]
        segmentation_bitmap_url = label["attributes"]["segmentation_bitmap"]["url"]
        url_extension = os.path.splitext(urlparse(segmentation_bitmap_url).path)[1]

        if not segmentation_bitmap_url:
            return None

        if self.caching_enabled:
            # segmentation_bitmap_filename = os.path.join(self.image_dir, '{}{}'.format(label['uuid'], url_extension))
            segmentation_bitmap_filename = os.path.join(
                self.image_dir,
                f"{sample_name}_label_{labelset}{url_extension}",
            )
            if not os.path.exists(segmentation_bitmap_filename):
                return load_label_bitmap_from_url(segmentation_bitmap_url, segmentation_bitmap_filename)
            else:
                return Image.open(segmentation_bitmap_filename)
        else:
            return load_label_bitmap_from_url(segmentation_bitmap_url)

    @property
    def categories(self) -> List[SegmentsDatasetCategory]:
        return TypeAdapter(List[SegmentsDatasetCategory]).validate_python(
            self.release["dataset"]["task_attributes"]["categories"],
        )
        # categories = {}
        # for category in self.release['dataset']['labelsets'][self.labelset]['attributes']['categories']:
        #     categories[category['id']] = category['name']
        # return categories

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample: Dict[str, Any] = self.samples[index]

        if self.task_type in [
            "pointcloud-segmentation",
            "pointcloud-cuboid",
            "image-vector-sequence",
        ]:
            return sample

        # Load the image
        image, image_filename = None, None
        try:
            image, image_filename = self._load_image_from_cache(sample)
        except (KeyError, TypeError):
            logger.error(f"Something went wrong loading sample {sample['name']}: {sample}")

        item = {
            "uuid": sample["uuid"],
            "name": sample["name"],
            "file_name": image_filename,
            "image": image,
            "metadata": sample["metadata"],
        }

        # Segmentation bitmap
        if self.task_type == "segmentation-bitmap" or self.task_type == "segmentation-bitmap-highres":
            # Load the label
            try:
                label = sample["labels"][self.labelset]
                segmentation_bitmap = self._load_segmentation_bitmap_from_cache(sample, self.labelset)
                attributes = label["attributes"]
                annotations = attributes["annotations"]
                item.update(
                    {
                        "segmentation_bitmap": segmentation_bitmap,
                        "annotations": annotations,
                        "attributes": attributes,
                    }
                )
            except (KeyError, TypeError):
                item.update(
                    {
                        "segmentation_bitmap": None,
                        "annotations": None,
                        "attributes": None,
                    }
                )

        # Vector labels
        elif self.task_type == "vector" or self.task_type == "bboxes" or self.task_type == "keypoints":
            try:
                label = sample["labels"][self.labelset]
                attributes = label["attributes"]
                annotations = attributes["annotations"]
                item.update({"annotations": annotations, "attributes": attributes})
            except (KeyError, TypeError):
                item.update({"annotations": None, "attributes": None})

        else:
            raise ValueError("This task type is not yet supported.")

        #         # transform
        #         if self.transform:
        #             item = self.transform(item)

        return item
