from __future__ import annotations

import json
import logging
import re
from io import BytesIO
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Tuple, Union, cast
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
    ] = "coco-panoptic",
    id_increment: int = 0,
    **kwargs: Any,
) -> Optional[Union[Tuple[str, Optional[str]], Optional[str]]]:
    """Export a dataset to a different format.

    Args:
        dataset: A :class:`.SegmentsDataset`.
        export_folder: The folder to export the dataset to. Defaults to ``.``.
        export_format: The destination format. Defaults to ``coco-panoptic``.
        id_increment: Increment the category ids with this number. Defaults to ``0``. Ignored unless ``export_format`` is ``semantic`` or ``semantic-color``.
    Returns:
        TODO
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
    return None


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
