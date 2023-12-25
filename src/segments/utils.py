from __future__ import annotations

import copy
import json
import logging
import os
import random
import re
from collections import defaultdict
from io import BytesIO
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union, cast
from urllib.parse import urlparse

import numpy as np
import numpy.typing as npt
import requests
from PIL import ExifTags, Image
from segments import SegmentsClient
from segments.exceptions import AlreadyExistsError
from segments.typing import (
    XYZW,
    EgoPose,
    ExportFormat,
    PointcloudCuboidLabelAttributes,
    PointcloudSequenceSampleAttributes,
    TaskType,
)


# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
# https://stackoverflow.com/questions/61384752/how-to-type-hint-with-an-optional-import
if TYPE_CHECKING:
    import open3d as o3d
    from pyquaternion import Quaternion
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
COMPATIBLE_TASK_TYPES = {
    ExportFormat.COCO_PANOPTIC: {
        TaskType.SEGMENTATION_BITMAP,
        TaskType.SEGMENTATION_BITMAP_HIGHRES,
    },
    ExportFormat.COCO_INSTANCE: {
        TaskType.VECTOR,
        TaskType.BBOXES,
        TaskType.SEGMENTATION_BITMAP,
        TaskType.SEGMENTATION_BITMAP_HIGHRES,
    },
    ExportFormat.YOLO: {
        TaskType.VECTOR,
        TaskType.BBOXES,
        TaskType.IMAGE_VECTOR_SEQUENCE,
    },
    ExportFormat.INSTANCE: {
        TaskType.SEGMENTATION_BITMAP,
        TaskType.SEGMENTATION_BITMAP_HIGHRES,
    },
    ExportFormat.INSTANCE_COLOR: {
        TaskType.SEGMENTATION_BITMAP,
        TaskType.SEGMENTATION_BITMAP_HIGHRES,
    },
    ExportFormat.SEMANTIC: {
        TaskType.SEGMENTATION_BITMAP,
        TaskType.SEGMENTATION_BITMAP_HIGHRES,
    },
    ExportFormat.SEMANTIC_COLOR: {
        TaskType.SEGMENTATION_BITMAP,
        TaskType.SEGMENTATION_BITMAP_HIGHRES,
    },
    ExportFormat.POLYGON: {
        TaskType.SEGMENTATION_BITMAP,
        TaskType.SEGMENTATION_BITMAP_HIGHRES,
    },
}


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
    export_format: ExportFormat = ExportFormat.COCO_PANOPTIC,
    id_increment: int = 0,
    **kwargs: Any,
) -> Optional[Union[Tuple[str, Optional[str]], Optional[str]]]:
    """Export a dataset to a different format.

    +------------------+-------------------------------------------------------------+
    | Export format    | Supported dataset type                                      |
    +==================+=============================================================+
    | COCO panoptic    | ``segmentation-bitmap`` and ``segmentation-bitmap-highres`` |
    +------------------+-------------------------------------------------------------+
    | COCO instance    | ``segmentation-bitmap`` and ``segmentation-bitmap-highres`` |
    +------------------+-------------------------------------------------------------+
    | YOLO             | ``vector``, ``bboxes`` and ``image-vector-sequence``        |
    +------------------+-------------------------------------------------------------+
    | Instance         | ``segmentation-bitmap`` and ``segmentation-bitmap-highres`` |
    +------------------+-------------------------------------------------------------+
    | Colored instance | ``segmentation-bitmap`` and ``segmentation-bitmap-highres`` |
    +------------------+-------------------------------------------------------------+
    | Semantic         | ``segmentation-bitmap`` and ``segmentation-bitmap-highres`` |
    +------------------+-------------------------------------------------------------+
    | Colored semantic | ``segmentation-bitmap`` and ``segmentation-bitmap-highres`` |
    +------------------+-------------------------------------------------------------+
    | Polygon          | ``segmentation-bitmap`` and ``segmentation-bitmap-highres`` |
    +------------------+-------------------------------------------------------------+

    Example:

    .. code-block:: python

        # pip install segments-ai
        from segments import SegmentsClient, SegmentsDataset
        from segments.utils import export_dataset

        # Initialize a SegmentsDataset from the release file
        client = SegmentsClient('YOUR_API_KEY')
        release = client.get_release('jane/flowers', 'v1.0') # Alternatively: release = 'flowers-v1.0.json'
        dataset = SegmentsDataset(release, labelset='ground-truth', filter_by=['labeled', 'reviewed'])

        # Export to COCO panoptic format
        export_dataset(dataset, export_format='coco-panoptic')


    Alternatively, you can use the initialized :class:`.SegmentsDataset` to loop through the samples and labels, and visualize or process them in any way you please:

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
    if export_format == ExportFormat.COCO_PANOPTIC:
        if dataset.task_type not in COMPATIBLE_TASK_TYPES[export_format]:
            raise ValueError(
                "Only datasets of type 'segmentation-bitmap' and 'segmentation-bitmap-highres' can be exported to this format."
            )
        from .export import export_coco_panoptic

        return export_coco_panoptic(dataset, export_folder)
    elif export_format == ExportFormat.COCO_INSTANCE:
        if dataset.task_type not in COMPATIBLE_TASK_TYPES[export_format]:
            raise ValueError(
                "Only datasets of type 'segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes' and 'keypoints' can be exported to this format."
            )
        from .export import export_coco_instance

        return export_coco_instance(dataset, export_folder)
    elif export_format == ExportFormat.YOLO:
        if dataset.task_type not in COMPATIBLE_TASK_TYPES[export_format]:
            raise ValueError(
                'Only datasets of type "vector", "bboxes" and "image-vector-sequence" can be exported to this format.'
            )
        from .export import export_yolo

        return export_yolo(
            dataset,
            export_folder,
            image_width=kwargs.get("image_width", None),
            image_height=kwargs.get("image_height", None),
        )
    elif export_format in {
        ExportFormat.SEMANTIC_COLOR,
        ExportFormat.INSTANCE_COLOR,
        ExportFormat.SEMANTIC,
        ExportFormat.INSTANCE,
    }:
        if dataset.task_type not in COMPATIBLE_TASK_TYPES[export_format]:
            raise ValueError(
                "Only datasets of type 'segmentation-bitmap' and 'segmentation-bitmap-highres' can be exported to this format."
            )
        from .export import export_image

        return export_image(dataset, export_folder, export_format, id_increment)
    elif export_format == ExportFormat.POLYGON:
        if dataset.task_type not in COMPATIBLE_TASK_TYPES[export_format]:
            raise ValueError(
                'Only datasets of type "segmentation-bitmap" and "segmentation-bitmap-highres" can be exported to this format.'
            )
        from .export import export_polygon

        return export_polygon(dataset, export_folder)
    else:
        raise ValueError("Please choose a valid export_format.")


def load_image_from_url(url: str, save_filename: Optional[str] = None, s3_client: Optional[Any] = None) -> Image.Image:
    """Load an image from url.

    Args:
        url: The image url.
        save_filename: The filename to save to.
        s3_client: A boto3 S3 client, e.g. ``s3_client = boto3.client("s3")``. Needs to be provided if your images are in a private S3 bucket. Defaults to :obj:`None`.

    Returns:
        A PIL image.
    """
    if s3_client is not None:
        url_parsed = urlparse(url)
        regex = re.search(r"(.+).(s3|s3-accelerate).(.+).amazonaws.com", url_parsed.netloc)
        if regex:
            bucket = regex.group(1)

            if bucket == "segmentsai-prod":
                image = Image.open(BytesIO(session.get(url).content))
            else:
                # region_name = regex.group(2)
                key = url_parsed.path.lstrip("/")

                file_byte_string = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
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


def load_pointcloud_from_url(
    url: str, save_filename: Optional[str] = None, s3_client: Optional[Any] = None
) -> o3d.geometry.PointCloud:
    """Load a point cloud from a url.

    Args:
        url: The point cloud url.
        save_filename: The filename to save to.
        s3_client: A boto3 S3 client, e.g. ``s3_client = boto3.client("s3")``. Needs to be provided if your point clouds are in a private S3 bucket. Defaults to :obj:`None`.

    Returns:
        A point cloud.

    Raises:
        :exc:`ImportError`: If open3d is not installed (to install run ``pip install open3d``).
    """

    try:
        import open3d as o3d
    except ImportError as e:
        logger.error("Please install open3d first: pip install open3d")
        raise e

    def load_pointcloud_from_parsed_url(url: str) -> o3d.geometry.PointCloud:
        with NamedTemporaryFile(suffix=".pcd") as f:
            f.write(session.get(url).content)
            pointcloud = o3d.io.read_point_cloud(f.name)

        return pointcloud

    if s3_client is not None:
        url_parsed = urlparse(url)
        regex = re.search(r"(.+).(s3|s3-accelerate).(.+).amazonaws.com", url_parsed.netloc)
        if regex:
            bucket = regex.group(1)

            if bucket == "segmentsai-prod":
                pointcloud = load_pointcloud_from_parsed_url(url)
            else:
                key = url_parsed.path.lstrip("/")
                file_byte_string = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read()
                with NamedTemporaryFile(suffix=".pcd") as f:
                    f.write(file_byte_string)
                    pointcloud = o3d.io.read_point_cloud(f.name)
    else:
        pointcloud = load_pointcloud_from_parsed_url(url)
    if save_filename is not None:
        o3d.io.write_point_cloud(save_filename, pointcloud)

    return pointcloud


def load_label_bitmap_from_url(url: str, save_filename: Optional[str] = None) -> npt.NDArray[np.uint32]:
    """Load a label bitmap from url.

    Args:
        url: The label bitmap url.
        save_filename: The filename to save to.

    Returns:
        A :class:`numpy.ndarray` with :class:`numpy.uint32` ``dtype`` where each unique value represents an instance id.
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

    Returns:
        None

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
    filtered_annotations = filter(lambda dictionary: dictionary["image_id"] == image_id, polygons["annotations"])
    for annotation in filtered_annotations:
        annotations[annotation["category_id"]].extend(annotation["polygons"])

    # {category name: (polygons, color)}
    category_name_polygons_with_annotations = {
        category_name: (annotations[category_id], category_color)
        for category_id, (category_name, category_color) in categories.items()
        if annotations[category_id]
    }

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(25, 10))

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
                label=category_name if category_name not in used_category_names else None,
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
        path = os.path.join(output_path, f"exported_polygons_from_image_id_{image_id:04d}")
        plt.savefig(path, bbox_inches="tight")

    plt.show()


def cuboid_to_segmentation(
    pointcloud: npt.NDArray[np.float32],
    label_attributes: PointcloudCuboidLabelAttributes,
    ego_pose: Optional[EgoPose] = None,
) -> npt.NDArray[np.uint32]:
    """Convert a cuboid label to an instance segmentation label.

    Args:
        pointcloud: A point cloud of size Nx3.
        label_attributes: A cuboid label from a single frame interface or one frame from a sequence interface.

    Returns:
        An instance segmentation label of size Nx1 mapping each point cloud point to a cuboid instance.

    Raises:
        :exc:`ImportError`: If pyquaternion is not installed (to install run ``pip install pyquaternion``).
        :exc:`ImportError`: If open3d is not installed (to install run ``pip install open3d``).
    """

    try:
        from pyquaternion import Quaternion
    except ImportError as e:
        logger.error("Please install pyquaternion first: pip install pyquaternion")
        raise e

    try:
        import open3d as o3d
    except ImportError as e:
        logger.error("Please install open3d first: pip install open3d")
        raise e

    # check dimensions of input
    assert pointcloud.shape[1] == 3, "Pointcloud must have shape (N, 3)"
    assert label_attributes.annotations, "Label must have annotations"

    # create cuboids
    cuboids = {}
    for annotation in label_attributes.annotations:
        center = np.array([annotation.position.x, annotation.position.y, annotation.position.z])
        extent = np.array([annotation.dimensions.x, annotation.dimensions.y, annotation.dimensions.z])
        if annotation.rotation:
            rotation = o3d.geometry.get_rotation_matrix_from_quaternion(
                np.array(
                    [
                        annotation.rotation.qx,
                        annotation.rotation.qy,
                        annotation.rotation.qz,
                        annotation.rotation.qw,
                    ]
                )
            )
        else:
            rotation = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, annotation.yaw))

        # create cuboid
        cuboid = o3d.geometry.OrientedBoundingBox(center=center, extent=extent, R=rotation)

        cuboids[annotation.id] = cuboid

        # transform cuboids from world to lidar coordinate frame
        transformation = np.eye(4)
        if ego_pose and ego_pose.position:
            pos = ego_pose.position
            transformation[:3, 3] = np.array([pos.x, pos.y, pos.z])
        if ego_pose and ego_pose.heading:
            rot = ego_pose.heading
            rotq = Quaternion(x=rot.qx, y=rot.qy, z=rot.qz, w=rot.qw).inverse.transformation_matrix
            transformation[:3, :3] = rotq[:3, :3]
        # tranform = rotate + translate (bug: transform not defined for OrientedBoundingBox)
        cuboid.translate(-transformation[:3, 3])
        cuboid.rotate(transformation[:3, :3])

    # map each point to a cuboid instance
    result = np.zeros(pointcloud.shape[0], dtype=np.uint32)
    pointcloud = o3d.utility.Vector3dVector(pointcloud)
    for id, cuboid in cuboids.items():
        # get points inside cuboid
        inside = cuboid.get_point_indices_within_bounding_box(pointcloud)
        result[inside] = id

    return result


def array_to_pcd(
    positions: npt.NDArray[np.float32],
    output_path: str,
    intensity: Optional[npt.NDArray[np.float32]] = None,
    rgb: Optional[npt.NDArray[np.float32]] = None,
    compressed: bool = False,
    write_ascii: bool = True,
) -> None:
    """Convert a numpy array to a pcd file.

    Args:
        positions: Array of xyz points (Nx3 shape).
        output_path: Path to write the pcd.
        intensity: Optional array of intensity values (Nx1 shape).
        rgb: Optional array of rgb values (Nx3 shape) where red, green and blue are values between 0 and 255 or 0 and 1.
        compressed: If the pcd should be compressed. Defaults to :obj:`False`.
        write_ascii: If the pcd should be written in ascii format. Defaults to :obj:`True`.

    Returns:
        None

    Raises:
        :exc:`ImportError`: If open3d is not installed (to install run ``pip install open3d``).
        :exc:`AssertionError`: If the positions array does not have shape (N, 3).
        :exc:`AssertionError`: If the intensity array does not have shape (N, 1) or (N,).
        :exc:`AssertionError`: If the rgb array does not have shape (N, 3).
        :exc:`AssertionError`: If the intensity array does not have the same length as the positions array.
        :exc:`AssertionError`: If the rgb array does not have the same length as the positions array.
    """

    try:
        import open3d as o3d
    except ImportError as e:
        logger.error("Please install open3d first: pip install open3d")
        raise e

    assert positions.shape[1] == 3, f"Positions must have shape (N, 3) but has shape {positions.shape}"

    # cast to float32
    positions = positions.astype(np.float32)
    intensity = intensity.astype(np.float32) if intensity is not None else None
    rgb = rgb.astype(np.float32) if rgb is not None else None

    device = o3d.core.Device("CPU:0")
    dtype = o3d.core.float32
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point["positions"] = o3d.core.Tensor(positions, dtype, device)
    N = positions.shape[0]

    if intensity is not None:
        assert intensity.shape in (
            (N, 1),
            (N,),
        ), f"Intensity must have shape ({N}, 1) or ({N},) but has shape {intensity.shape}"
        if len(intensity.shape) == 1:
            intensity = intensity.reshape(-1, 1)

        pcd.point["intensity"] = o3d.core.Tensor(intensity, dtype, device)

    if rgb is not None:
        assert rgb.shape == (N, 3), f"RGB must have shape ({N}, 3) but has shape {rgb.shape}"

        # check rgb encoding (0-255 or 0-1)
        if np.max(rgb) > 1:
            rgb /= 255.0  # map 0-255 to 0-1 (open3d expects rgb values between 0 and 1)

        pcd.point["colors"] = o3d.core.Tensor(rgb, dtype, device)

    o3d.t.io.write_point_cloud(
        output_path,
        pcd,
        compressed=compressed,
        write_ascii=write_ascii,
        print_progress=True,
    )


def ply_to_pcd(ply_file: str, compressed: bool = False, write_ascii: bool = True) -> None:
    """Convert a .ply file to a .pcd file.

    Args:
        ply_file: The path to the .ply file.
        compressed: If the pcd should be compressed. Defaults to :obj:`False`.
        write_ascii: If the pcd should be written in ascii format. Defaults to :obj:`True`.

    Returns:
        None

    Raises:
        :exc:`ImportError`: If plyfile is not installed (to install run ``pip install plyfile``).
        :exc:`KeyError`: If the positions are not found in the ply file (expected colum names are ``x``, ``y`` and ``z``).
    """

    try:
        from plyfile import PlyData
    except ImportError as e:
        logger.error("Please install plyfile first: pip install plyfile")
        raise e

    with open(ply_file, "rb") as f:
        ply = PlyData.read(f)

    try:
        positions = np.stack((ply["vertex"]["x"], ply["vertex"]["y"], ply["vertex"]["z"]), axis=-1)
    except KeyError:
        raise KeyError("Could not find the positions in the ply file.")

    try:
        intensity = np.array(ply["vertex"]["intensity"]).reshape(-1, 1)
    except KeyError:
        try:
            intensity = np.array(ply["vertex"]["i"]).reshape(-1, 1)
        except KeyError:
            intensity = None

    try:
        rgb = np.stack((ply["vertex"]["r"], ply["vertex"]["g"], ply["vertex"]["b"]), axis=-1)
    except KeyError:
        try:
            rgb = np.stack(
                (ply["vertex"]["red"], ply["vertex"]["green"], ply["vertex"]["blue"]),
                axis=-1,
            )
        except KeyError:
            rgb = None

    pcd_path = ply_file.replace(".ply", ".pcd")
    # prefer RGB over intensity (tiled point cloud does not support both)
    intensity = intensity if rgb is None else None
    array_to_pcd(
        positions,
        pcd_path,
        intensity=intensity,
        rgb=rgb,
        compressed=compressed,
        write_ascii=write_ascii,
    )


def sample_pcd(pcd_path: str, points: int = 500_000, output_path: Optional[str] = None) -> None:
    """Sample a point cloud to a given number of points.

    Args:
        pcd_path: The path to the point cloud.
        points: The number of points to sample. Defaults to ``500_000``.
        output_path: The path to save the sampled point cloud to. Defaults to :obj:`None`.

    Returns:
        None

    Raises:
        :exc:`ImportError`: If open3d is not installed (to install run ``pip install open3d``).
    """

    try:
        import open3d as o3d
    except ImportError as e:
        logger.error("Please install open3d first: pip install open3d")
        raise e

    if output_path is None:
        output_path = output_path.replace(".pcd", "_sampled.pcd")

    pcd = o3d.io.read_point_cloud(pcd_path)
    # open3d expects a step size (not a number of points)
    points_step_size = len(pcd.points) // points
    pcd = pcd.uniform_down_sample(points_step_size)
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False, compressed=True, print_progress=True)


def find_camera_rotation(client: SegmentsClient, dataset_identifier: str) -> Quaternion:
    """Find the correct camera rotation by trying all possibilities.

    Args:
        client: A Segments client.
        dataset_identifier: The dataset identifier.

    Returns:
        A :class:`pyquaternion.Quaternion` representing the correct rotation.

    Raises:
        :exc:`ImportError`: If pyquaternion is not installed (to install run ``pip install pyquaternion``).
        :exc:`ValueError`: If the dataset is not a point cloud sequence dataset.
        :exc:`ValueError`: If the user answers neither 'y(es)' nor 'n(o)' (case insensitive).
        :exc:`ValueError`: If the correct rotation is not found.
        :exc:`AlreadyExistsError`: If the cloned dataset already exists.
    """

    try:
        from pyquaternion import Quaternion
    except ImportError as e:
        logger.error("Please install pyquaternion first: pip install pyquaternion")
        raise e

    # 6 options for x axis, 4 options for y axis and 1 option for z axis
    X_AXIS_ROTATIONS = [
        Quaternion(axis=[0, 0, 1], angle=0),
        Quaternion(axis=[0, 0, 1], angle=np.pi / 2),
        Quaternion(axis=[0, 0, 1], angle=np.pi),
        Quaternion(axis=[0, 0, 1], angle=3 * np.pi / 2),
        Quaternion(axis=[0, 1, 0], angle=np.pi / 2),
        Quaternion(axis=[0, 1, 0], angle=3 * np.pi / 2),
    ]
    Y_AXIS_ROTATIONS = [
        Quaternion(axis=[1, 0, 0], angle=0),
        Quaternion(axis=[1, 0, 0], angle=np.pi / 2),
        Quaternion(axis=[1, 0, 0], angle=np.pi),
        Quaternion(axis=[1, 0, 0], angle=3 * np.pi / 2),
    ]
    # inverse rotation is possible
    total_rotation_options = len(X_AXIS_ROTATIONS) * len(Y_AXIS_ROTATIONS) * 2

    # clone dataset
    cloned_dataset_owner, cloned_dataset_name = (
        dataset_identifier.split("/")[0],
        f"{dataset_identifier.split('/')[-1]}-find-camera-rotation",
    )
    try:
        cloned_dataset = client.clone_dataset(dataset_identifier, cloned_dataset_name)
    except AlreadyExistsError:
        client.delete_dataset(f"{cloned_dataset_owner}/{cloned_dataset_name}")
        cloned_dataset = client.clone_dataset(dataset_identifier, cloned_dataset_name)

    # get (cloned) samples
    samples = client.get_samples(dataset_identifier)
    cloned_samples = client.get_samples(cloned_dataset.full_name)
    if not isinstance(samples[0].attributes, PointcloudSequenceSampleAttributes):
        raise ValueError(
            "Brute force camera rotations only works for point cloud sequence datasets. Reach out to support@segments.ai and arnaud@segments.ai if you are interested in this functionality for other dataset types."
        )

    # rotate images
    for xi, x_rot in enumerate(X_AXIS_ROTATIONS):
        for yi, y_rot in enumerate(Y_AXIS_ROTATIONS):
            for invert in [False, True]:
                for sample, cloned_sample in zip(samples, cloned_samples):
                    for frame, cloned_frame in zip(sample.attributes.frames, cloned_sample.attributes.frames):
                        for image, cloned_image in zip(frame.images, cloned_frame.images):
                            if not image.extrinsics or not image.extrinsics.rotation:
                                continue

                            rot = image.extrinsics.rotation
                            rot_q = Quaternion(x=rot.qx, y=rot.qy, z=rot.qz, w=rot.qw)

                            if invert:
                                rot_q = rot_q.inverse

                            new_rot_q = x_rot * y_rot * rot_q
                            new_rot = XYZW(
                                qx=new_rot_q.x,
                                qy=new_rot_q.y,
                                qz=new_rot_q.z,
                                qw=new_rot_q.w,
                            )
                            cloned_image.extrinsics.rotation = new_rot
                    client.update_sample(cloned_sample.uuid, attributes=cloned_sample.attributes)

                offset, y_offset, x_offset = 1, 2, 2 * len(Y_AXIS_ROTATIONS)
                rotation_option = xi * x_offset + yi * y_offset + invert + offset

                rotation_OK = input(
                    f"Correct image rotation in {cloned_dataset.full_name} (rotation {rotation_option}/{total_rotation_options})? [y/n]"
                )
                if rotation_OK.lower() in ["y", "yes"]:
                    client.delete_dataset(cloned_dataset.full_name)
                    return (x_rot * y_rot).inverse if invert else x_rot * y_rot
                elif rotation_OK.lower() not in ["n", "no"]:
                    raise ValueError(f"Please enter 'y(es)' or 'n(o)' (case insensitive) not {rotation_OK}.")

    raise ValueError("Correct rotation not found.")
