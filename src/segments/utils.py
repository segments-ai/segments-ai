from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
from collections import defaultdict
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
import requests
from PIL import ExifTags, Image
from typing_extensions import Literal

# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from segments.dataset import SegmentsDataset
    from segments.typing import Release


#############
# Variables #
#############
session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount("http://", adapter)
session.mount("https://", adapter)
logger = logging.getLogger(__name__)


def bitmap2file(
    bitmap: npt.NDArray[np.uint32],
    is_segmentation_bitmap: bool = True,
) -> BytesIO:
    """Convert a label bitmap to a file with the proper format.

    Args:
        bitmap: A :class:`numpy.ndarray` with :class:`numpy.uint32` dtype where each unique value represents an instance id.
        is_segmentation_bitmap: If this is a segmentation bitmap. Defaults to :obj:`True`.
    Returns:
        A file object.
    Raises:
        :exc:`ValueError`: If the ``dtype`` is not :class:`np.uint32` or :class:`np.uint8`.
        :exc:`ValueError`: If the bitmap is not a segmentation bitmap.
    """

    # Convert bitmap to np.uint32, if it is not already
    if bitmap.dtype == "uint32":
        pass
    elif bitmap.dtype == "uint8":
        bitmap = np.uint32(bitmap)
    else:
        raise ValueError("Only np.ndarrays with np.uint32 dtype can be used.")

    if is_segmentation_bitmap:
        bitmap2 = np.copy(bitmap)
        bitmap2 = bitmap2[:, :, None].view(np.uint8)
        bitmap2[:, :, 3] = 255
    else:
        raise ValueError("Only segmentation bitmaps can be used.")

    f = BytesIO()
    Image.fromarray(bitmap2).save(f, "PNG")
    f.seek(0)
    return f


def get_semantic_bitmap(
    instance_bitmap: Optional[npt.NDArray[np.uint32]] = None,
    annotations: Optional[Dict[str, Any]] = None,
    id_increment: int = 0,
) -> Optional[npt.NDArray[np.uint32]]:
    """Convert an instance bitmap and annotations dict into a segmentation bitmap.

    Args:
        instance_bitmap: A :class:`numpy.ndarray` with :class:`numpy.uint32` ``dtype`` where each unique value represents an instance id. Defaults to :obj:`None`.
        annotations: An annotations dictionary. Defaults to :obj:`None`.
        id_increment: Increment the category ids with this number. Defaults to ``0``.
    Returns:
        An array here each unique value represents a category id.
    """

    if instance_bitmap is None or annotations is None:
        return None

    instance2semantic = [0] * (max([a["id"] for a in annotations], default=0) + 1)
    for annotation in annotations:
        instance2semantic[annotation["id"]] = annotation["category_id"] + id_increment
    instance2semantic = np.array(instance2semantic)

    semantic_label = instance2semantic[np.array(instance_bitmap, np.uint32)]
    return semantic_label


def export_dataset(
    dataset: SegmentsDataset,
    export_folder: str = ".",
    export_format: Literal[
        "coco-panoptic",
        "coco-instance",
        "yolo",
        "instance",
        "instance-color",
        "semantic",
        "semantic-color",
        "polygon",
    ] = "coco-panoptic",
    id_increment: int = 0,
    **kwargs: Any,
) -> Optional[Union[Tuple[str, Optional[str]], Optional[str]]]:
    """Export a dataset to a different format.

    +------------------+----------------------------------------------------------------------------------------------------+
    | Export format    | Supported dataset type                                                                             |
    +==================+====================================================================================================+
    | COCO panoptic    | ``segmentation-bitmap`` and ``segmentation-bitmap-highres``                                        |
    +------------------+----------------------------------------------------------------------------------------------------+
    | COCO instance    | ``segmentation-bitmap`` and ``segmentation-bitmap-highres``                                        |
    +------------------+----------------------------------------------------------------------------------------------------+
    | YOLO             | ``segmentation-bitmap``, ``segmentation-bitmap-highres``, ``vector``, ``bboxes`` and ``keypoints`` |
    +------------------+----------------------------------------------------------------------------------------------------+
    | Instance         | ``segmentation-bitmap`` and ``segmentation-bitmap-highres``                                        |
    +------------------+----------------------------------------------------------------------------------------------------+
    | Colored instance | ``segmentation-bitmap`` and ``segmentation-bitmap-highres``                                        |
    +------------------+----------------------------------------------------------------------------------------------------+
    | Semantic         | ``segmentation-bitmap`` and ``segmentation-bitmap-highres``                                        |
    +------------------+----------------------------------------------------------------------------------------------------+
    | Colored semantic | ``segmentation-bitmap`` and ``segmentation-bitmap-highres``                                        |
    +------------------+----------------------------------------------------------------------------------------------------+
    | Polygon          | ``segmentation-bitmap`` and ``segmentation-bitmap-highres``                                        |
    +------------------+----------------------------------------------------------------------------------------------------+

    Args:
        dataset: A :class:`.SegmentsDataset`.
        export_folder: The folder to export the dataset to. Defaults to ``.``.
        export_format: The destination format. Defaults to ``coco-panoptic``.
        id_increment: Increment the category ids with this number. Defaults to ``0``. Ignored unless ``export_format`` is ``semantic`` or ``semantic-color``.
    Returns:
        Returns the file name and the image directory name (for COCO panoptic, COCO instance, YOLO and polygon), or returns the export folder name (for (colored) instance and (colored) panoptic).
    Raises:
        :exc:`ImportError`: If scikit image is not installed (to install run ``pip install scikit-image``).
        :exc:`ValueError`: If an unvalid ``export_format`` is used.
    """

    try:
        import skimage  # noqa: F401
    except ImportError as e:
        logger.error("Please install scikit-image first: pip install scikit-image.")
        raise e

    print("Exporting dataset. This may take a while...")
    if export_format == "coco-panoptic":
        if dataset.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            raise ValueError(
                "Only datasets of type 'segmentation-bitmap' and 'segmentation-bitmap-highres' can be exported to this format."
            )
        from .export import export_coco_panoptic

        return export_coco_panoptic(dataset, export_folder)
    elif export_format == "coco-instance":
        if dataset.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            raise ValueError(
                "Only datasets of type 'segmentation-bitmap' and 'segmentation-bitmap-highres' can be exported to this format."
            )
        from .export import export_coco_instance

        return export_coco_instance(dataset, export_folder)
    elif export_format == "yolo":
        if dataset.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
            "vector",
            "bboxes",
            "keypoints",
        ]:
            raise ValueError(
                'Only datasets of type "segmentation-bitmap", "segmentation-bitmap-highres", "vector", "bboxes" and "keypoints" can be exported to this format.'
            )
        from .export import export_yolo

        return export_yolo(
            dataset,
            export_folder,
            image_width=kwargs.get("image_width", None),
            image_height=kwargs.get("image_height", None),
        )
    elif export_format in ["semantic-color", "instance-color", "semantic", "instance"]:
        if dataset.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            raise ValueError(
                "Only datasets of type 'segmentation-bitmap' and 'segmentation-bitmap-highres' can be exported to this format."
            )
        from .export import export_image

        return export_image(dataset, export_folder, export_format, id_increment)
    elif export_format == "polygon":
        if dataset.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            raise ValueError(
                'Only datasets of type "segmentation-bitmap" and "segmentation-bitmap-highres" can be exported to this format.'
            )
        from .export import export_polygon

        return export_polygon(dataset, export_folder)
    else:
        raise ValueError("Please choose a valid export_format.")


def load_image_from_url(
    url: str, save_filename: Optional[str] = None, s3_client: Optional[Any] = None
) -> Image.Image:
    """Load an image from url.

    Args:
        url: The image url.
        save_filename: The filename to save to.
        s3_client: A boto3 S3 client, e.g. ``s3_client = boto3.client("s3")``. Needs to be provided if your images are in a private S3 bucket. Defaults to :obj:`None`.
    """
    if s3_client is not None:
        url_parsed = urlparse(url)
        regex = re.search(
            r"(.+).(s3|s3-accelerate).(.+).amazonaws.com", url_parsed.netloc
        )
        if regex:
            bucket = regex.group(1)

            if bucket == "segmentsai-prod":
                image = Image.open(BytesIO(session.get(url).content))
            else:
                # region_name = regex.group(2)
                key = url_parsed.path.lstrip("/")

                file_byte_string = s3_client.get_object(Bucket=bucket, Key=key)[
                    "Body"
                ].read()
                image = Image.open(BytesIO(file_byte_string))
    else:
        image = Image.open(BytesIO(session.get(url).content))
        # urllib.request.urlretrieve(url, save_filename)

    if save_filename is not None:
        if "exif" in image.info:
            image.save(save_filename, exif=image.info["exif"])
        else:
            image.save(save_filename)

    return image


def load_label_bitmap_from_url(
    url: str, save_filename: Optional[str] = None
) -> npt.NDArray[np.uint32]:
    """Load a label bitmap from url.

    Args:
        url: The label bitmap url.
        save_filename: The filename to save to.
    """

    def extract_bitmap(
        bitmap: Image.Image,
    ) -> npt.NDArray[np.uint32]:

        bitmap_array = np.array(bitmap)
        bitmap_array[:, :, 3] = 0
        bitmap_array = bitmap_array.view(np.uint32).squeeze(2)
        return bitmap_array

    bitmap = Image.open(BytesIO(session.get(url).content))
    bitmap_array = extract_bitmap(bitmap)

    if save_filename:
        Image.fromarray(bitmap_array).save(save_filename)

    return bitmap_array


def load_release(release: Release) -> Any:
    """Load JSON from Segments release.

    Args:
        release: A Segments release.
    Returns:
        A JSON with the release labels.
    """
    release_file = cast(str, release.attributes.url)  # TODO Fix in the backend.
    content = requests.get(release_file)
    return json.loads(content.content)


def handle_exif_rotation(image: Image.Image) -> Image.Image:
    """Handle the exif rotation of a PIL image.

    Args:
        image: A PIL image.
    Returns:
        A rotated PIL image.
    """

    def get_key_by_value(dictionary: Mapping[int, str], value: str) -> int:
        for k, v in dictionary.items():
            if v == value:
                return k
        raise ValueError(f"No such value {value}.")

    try:
        orientation = get_key_by_value(ExifTags.TAGS, "Orientation")
        exif = dict(image.getexif().items())
        if exif[orientation] == 3:
            image = image.transpose(Image.ROTATE_180)
        elif exif[orientation] == 6:
            image = image.transpose(Image.ROTATE_270)
        elif exif[orientation] == 8:
            image = image.transpose(Image.ROTATE_90)
        return image
    except (AttributeError, KeyError, IndexError, ValueError):
        return image


def show_polygons(
    image_directory_path: str,
    image_id: int,
    exported_polygons_path: str,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> None:
    """Show the exported contours of a segmented image (i.e., resulting from :func:`.export_dataset` with polygon export format).

    Args:
        image_directory_path: The image directory path.
        image_id: The image id (this can be found in the exported polygons JSON file).
        exported_polygons_path: The exported polygons path.
        seed: The seed used to generate random colors. Defaults to ``0``.
        output_path: The directory to save the plot to. Defaults to :obj:`None`.
    Raises:
        :exc:`ImportError`: If matplotlib is not installed.
    """

    try:
        from matplotlib import image
        from matplotlib import pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError as e:
        logger.error("Please install matplotlib first: pip install matplotlib.")
        raise e

    def find_image_name(images: List[Dict[str, Any]], image_id: int) -> str:
        for image in images:
            if image["id"] == image_id:
                return cast(str, image["file_name"])
        raise KeyError("Cannot find the image id. Please provide a valid id.")

    def get_random_color() -> Tuple[float, float, float]:
        return (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

    def normalize(color: List[int]) -> Tuple[float, float, float]:
        """Transform a color from 0-255 range to 0-1 range and from a list to a tuple, e.g., [255, 0, 123] to (1, 0, 0.5)."""
        return (color[0] / 255, color[1] / 255, color[2] / 255)

    random.seed(seed)

    with open(exported_polygons_path, "r") as f:
        polygons = json.load(f)

    image_name = find_image_name(polygons["images"], image_id)
    image = image.imread(f"{image_directory_path}/{image_name}")

    # {category id: (category name, color)}
    categories = {
        category["id"]: (
            category["name"],
            normalize(category["color"]) if category["color"] else get_random_color(),
        )
        for category in polygons["categories"]
    }

    # {category id: polygons}
    annotations = defaultdict(list)
    filtered_annotations = filter(
        lambda dictionary: dictionary["image_id"] == image_id, polygons["annotations"]
    )
    for annotation in filtered_annotations:
        annotations[annotation["category_id"]].extend(annotation["polygons"])

    # {category name: (polygons, color)}
    category_name_polygons_with_annotations = {
        category_name: (annotations[category_id], category_color)
        for category_id, (category_name, category_color) in categories.items()
        if annotations[category_id]
    }

    fig, (ax1, ax2, ax3) = plt.subplots(
        nrows=1, ncols=3, sharex=True, sharey=True, figsize=(25, 10)
    )

    used_category_names = set()
    for category_name, (
        polygons,
        color,
    ) in category_name_polygons_with_annotations.items():
        for p in polygons:
            polygon = Polygon(
                xy=np.asarray(p).reshape(-1, 2),
                facecolor=color,
                edgecolor=color,
                label=category_name
                if category_name not in used_category_names
                else None,
                closed=True,
                alpha=0.5,
            )

            used_category_names.add(category_name)
            polygon_copy = copy.deepcopy(polygon)
            polygon_copy.set_label(None)

            ax1.add_patch(polygon)
            # An Artist, container or primitive, cannot be contained in multiple containers, which is consistent with the fact that each Artist holds the parent container as a bare object, not in a list.
            ax2.add_patch(polygon_copy)

    # Ax 2
    # ax2.axis("off")
    ax2.set_title("Both")
    ax2.imshow(image)
    ax2.set_xlabel("Width (pixels)")

    # Ax 1 (uses the aspect ratio of the image in axes 2)
    # ax1.axis("off")
    ax1.set_title("Label")
    # ax1.imshow(image)
    # https://stackoverflow.com/a/44655020
    aspect = np.diff(ax1.get_xlim())[0] / np.diff(ax1.get_ylim())[0]
    aspect /= np.diff(ax2.get_xlim())[0] / np.diff(ax2.get_ylim())[0]
    aspect = np.abs(aspect)
    ax1.set_aspect(aspect)
    ax1.set_xlabel("Width (pixels)")
    ax1.set_ylabel("Height (pixels)")

    # Ax 3
    # ax3.axis("off")
    ax3.set_title("Image")
    ax3.imshow(image)
    ax3.set_xlabel("Width (pixels)")

    fig.legend()

    if output_path:
        path = os.path.join(
            output_path, f"exported_polygons_from_image_id_{image_id:04d}"
        )
        plt.savefig(path, bbox_inches="tight")

    plt.show()
