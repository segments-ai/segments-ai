import json
from io import BytesIO
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np  # import numpy.typing as npt
import requests
from PIL import ExifTags, Image
from segments.typing import Dataset, Release
from typing_extensions import Literal

session = requests.Session()
adapter = requests.adapters.HTTPAdapter(max_retries=3)
session.mount("http://", adapter)
session.mount("https://", adapter)


def bitmap2file(
    bitmap: Any,  # npt.NDArray[np.uint32],
    is_segmentation_bitmap: bool = True,
) -> BytesIO:
    """Convert a label bitmap to a file with the proper format.

    Args:
        bitmap (np.uint32): A numpy array where each unique value represents an instance id.

    Returns:
        A file object.
    """

    # Convert bitmap to np.uint32, if it is not already
    if bitmap.dtype == "uint32":
        pass
    elif bitmap.dtype == "uint8":
        bitmap = np.uint32(bitmap)
    else:
        assert False

    if is_segmentation_bitmap:
        bitmap2 = np.copy(bitmap)
        bitmap2 = bitmap2[:, :, None].view(np.uint8)
        bitmap2[:, :, 3] = 255
    else:
        assert False

    f = BytesIO()
    Image.fromarray(bitmap2).save(f, "PNG")
    f.seek(0)
    return f


def get_semantic_bitmap(
    instance_bitmap: Any,  # npt.NDArray[np.uint32],
    annotations: Dict[str, Any],
    id_increment: int = 1,
) -> Any:  # Optional[npt.NDArray[np.uint32]]:
    """Convert an instance bitmap and annotations dict into a segmentation bitmap.

    Args:
        instance_bitmap (np.uint32): A numpy array where each unique value represents an instance id.
        annotations: An annotations dictionary.
        id_increment: Increment the category ids with this number. Defaults to 1.

    Returns:
        np.uint32: a numpy array where each unique value represents a category id.
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
    dataset: Dataset,
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
    id_increment: int = 1,
) -> Optional[Union[Tuple[str, str], str]]:
    """Export a dataset to a different format.

    Args:
        dataset: A dataset class, resulting from client.get_dataset().
        export_folder: The folder to export the dataset to. Defaults to '.'.
        export_format: The destination format. Defaults to 'coco-panoptic'.
        id_increment: Increment the category ids with this number. Defaults to 1. Ignored unless export_format is 'semantic' or 'semantic-color'.

    """

    print("Exporting dataset. This may take a while...")
    if export_format == "coco-panoptic":
        if dataset.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            raise ValueError(
                'Only datasets of type "segmentation-bitmap" and "segmentation-bitmap-highres" can be exported to this format.'
            )
        from .export import export_coco_panoptic

        return export_coco_panoptic(dataset, export_folder)
    elif export_format == "coco-instance":
        if dataset.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            raise ValueError(
                'Only datasets of type "segmentation-bitmap" and "segmentation-bitmap-highres" can be exported to this format.'
            )
        from .export import export_coco_instance

        return export_coco_instance(dataset, export_folder)
    elif export_format == "yolo":
        if dataset.task_type not in ["vector", "bboxes"]:
            raise ValueError(
                'Only datasets of type "vector" and "bboxes" can be exported to this format.'
            )
        from .export import export_yolo

        return export_yolo(dataset, export_folder)
    elif export_format in ["semantic-color", "instance-color", "semantic", "instance"]:
        if dataset.task_type not in [
            "segmentation-bitmap",
            "segmentation-bitmap-highres",
        ]:
            raise ValueError(
                'Only datasets of type "segmentation-bitmap" and "segmentation-bitmap-highres" can be exported to this format.'
            )
        from .export import export_image

        return export_image(dataset, export_folder, export_format, id_increment)
    else:
        print("Supported export formats: coco-panoptic, coco-instance, yolo")
        return


def load_image_from_url(url: str, save_filename: Optional[str] = None) -> Image.Image:
    """Load an image from url.

    Args:
        url: The image url.
        save_filename: The filename to save to.

    Returns:
        A PIL image.
    """
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
) -> Any:  # npt.NDArray[np.uint32]:
    """Load a label bitmap from url.

    Args:
        url: The label bitmap url.
        save_filename: The filename to save to.

    Returns:
        np.uint32: a numpy np.uint32 array.
    """

    def extract_bitmap(
        bitmap: Image.Image,
    ) -> Any:  # def extract_bitmap(bitmap: Image.Image) -> npt.NDArray[np.uint32]:

        bitmap = np.array(bitmap)
        bitmap[:, :, 3] = 0
        bitmap = bitmap.view(np.uint32).squeeze(2)
        return bitmap

    bitmap = Image.open(BytesIO(session.get(url).content))
    bitmap = extract_bitmap(bitmap)

    if save_filename is not None:
        Image.fromarray(bitmap).save(save_filename)

    return bitmap


def load_release(release: Release) -> Any:
    release_file = release.attributes.url
    content = requests.get(release_file)
    return json.loads(content.content)


def handle_exif_rotation(image: Image.Image) -> Image.Image:

    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == "Orientation":
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.transpose(Image.ROTATE_180)
        elif exif[orientation] == 6:
            image = image.transpose(Image.ROTATE_270)
        elif exif[orientation] == 8:
            image = image.transpose(Image.ROTATE_90)
        return image
    except (AttributeError, KeyError, IndexError):
        return image
