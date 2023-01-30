# https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch/#coco-dataset-format
from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
import numpy.typing as npt
from PIL import Image
from segments.typing import SegmentsDatasetCategory
from segments.utils import get_semantic_bitmap
from skimage import img_as_ubyte
from skimage.measure import regionprops
from tqdm import tqdm

# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
if TYPE_CHECKING:
    from segments.dataset import SegmentsDataset


#############
# Variables #
#############
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]
ColorMap = Union[List[RGBA], List[RGB]]
logger = logging.getLogger(__name__)
COLORMAP: ColorMap = [
    (0, 113, 188, 255),
    (216, 82, 24, 255),
    (236, 176, 31, 255),
    (125, 46, 141, 255),
    (118, 171, 47, 255),
    (76, 189, 237, 255),
    (161, 19, 46, 255),
    (255, 0, 0, 255),
    (255, 127, 0, 255),
    (190, 190, 0, 255),
    (0, 255, 0, 255),
    (0, 0, 255, 255),
    (170, 0, 255, 255),
    (84, 84, 0, 255),
    (84, 170, 0, 255),
    (84, 255, 0, 255),
    (170, 84, 0, 255),
    (170, 170, 0, 255),
    (170, 255, 0, 255),
    (255, 84, 0, 255),
    (255, 170, 0, 255),
    (255, 255, 0, 255),
    (0, 84, 127, 255),
    (0, 170, 127, 255),
    (0, 255, 127, 255),
    (84, 0, 127, 255),
    (84, 84, 127, 255),
    (84, 170, 127, 255),
    (84, 255, 127, 255),
    (170, 0, 127, 255),
    (170, 84, 127, 255),
    (170, 170, 127, 255),
    (170, 255, 127, 255),
    (255, 0, 127, 255),
    (255, 84, 127, 255),
    (255, 170, 127, 255),
]


##################
# Helper methods #
##################
# https://github.com/cocodataset/panopticapi/blob/master/panopticapi/utils.py
class IdGenerator:
    """
    The class is designed to generate unique IDs that have meaningful RGB encoding.
    Given semantic category unique ID will be generated and its RGB encoding will
    have color close to the predefined semantic category color.
    The RGB encoding used is ID = R * 256 * G + 256 * 256 + B.
    Class constructor takes dictionary ``{id: category_info}``, where all semantic
    class ids are presented and ``category_info`` record is a dict with fields
    ``isthing`` and ``color``
    """

    def __init__(self, categories: Dict[int, SegmentsDatasetCategory]):
        self.taken_colors: Set[RGB] = set()
        self.taken_colors.add((0, 0, 0))
        self.categories = categories
        for category in self.categories.values():
            if category.isthing == 0:
                self.taken_colors.add(category.color)

    def get_color(self, cat_id: int) -> RGB:
        def random_color(base: RGB, max_dist: int = 30) -> RGB:
            new_color: npt.NDArray[Any] = base + np.random.randint(
                low=-max_dist, high=max_dist + 1, size=3
            )
            rgb = tuple(np.maximum(0, np.minimum(255, new_color)))
            return cast(RGB, rgb)

        category = self.categories[cat_id]
        if category.isthing == 0:
            return category.color
        base_color = category.color
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color)
                if color not in self.taken_colors:
                    self.taken_colors.add(color)
                    return color

    def get_id(self, cat_id: int) -> int:
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id: int) -> Tuple[int, RGB]:
        color = self.get_color(cat_id)
        return rgb2id(color), color


def rgb2id(color: Union[npt.NDArray[Any], RGB]) -> Union[npt.NDArray[Any], int]:
    """Convert rgb to an id.

    Args:
        color: An RGB value.
    Returns:
        The id.
    """
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        color_id = color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
        return cast(npt.NDArray[Any], color_id)
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map: npt.NDArray[Any]) -> Union[npt.NDArray[Any], RGB]:
    """Convert a color id to an rgb.

    Args:
        id_map: An id map.
    Returns:
        An rgb.
    """
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map: npt.NDArray[Any] = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return tuple(color)


def get_color(id: int) -> RGB:
    id = id % len(COLORMAP)
    return COLORMAP[id][0:3]


def colorize(
    img: npt.NDArray[Any], colormap: Optional[ColorMap] = None
) -> npt.NDArray[Any]:

    indices = np.unique(img)
    indices = indices[indices != 0]

    colored_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)

    for id in indices:
        mask = img == id
        if colormap:
            color = colormap[id - 1]
        else:
            color = get_color(id - 1)
        colored_img[mask] = color

    return colored_img


def get_bbox(binary_mask: npt.NDArray[Any]) -> Union[Tuple[int, int, int, int], bool]:
    """Returns the bounding box of the binary mask (if one is found, otherwise returns False)

    Args:
        binary_mask: The binary mask as a numpy array.
    Returns:
        The bounding box or False if no bounding box is found.
    """

    regions = regionprops(np.uint8(binary_mask))
    if len(regions) == 1:
        bbox = regions[0].bbox
        return cast(Tuple[int, int, int, int], bbox)

    return False


##################
# Export methods #
##################
def export_coco_instance(
    dataset: SegmentsDataset, export_folder: str
) -> Tuple[str, Optional[str]]:
    """Export a Segments dataset as a coco instance.

    Args:
        dataset: A :class:`.SegmentsDataset`.
        export_folder: The output directory.
    Returns:
        Returns the file name and the image directory name.
    Raises:
        :exc:`ImportError`: If pycocotools is not installed.
    """
    try:
        from pycocotools import mask as pctmask
    except ImportError as e:
        logger.error(
            "Please install pycocotools first: pip install pycocotools. Or on Windows: pip install pycocotools-windows"
        )
        raise e

    # Create export folder
    os.makedirs(export_folder, exist_ok=True)

    info = {
        "description": dataset.release["dataset"]["name"],
        "version": dataset.release["name"],
        # 'year': 2022,
    }

    categories = dataset.categories
    task_type = dataset.task_type
    # for i, category in enumerate(dataset.project_info['label_taxonomy']):
    #     categories.append({
    #         'id': i+1,
    #         'supercategory': 'object',
    #         'name': category
    #     })

    images = []
    annotations = []

    annotation_id = 1
    for i in tqdm(range(len(dataset)), total=len(dataset), colour="#FF9900"):
        sample = dataset[i]

        if sample["annotations"] is None:
            continue

        image_id = i + 1
        images.append(
            {
                "id": image_id,
                "file_name": sample["file_name"],
                "height": sample["image"].size[1] if sample["image"] else None,
                "width": sample["image"].size[0] if sample["image"] else None,
            }
        )

        if (
            task_type == "segmentation-bitmap"
            or task_type == "segmentation-bitmap-highres"
        ):
            # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
            regions = regionprops(np.array(sample["segmentation_bitmap"], np.uint32))
            regions = {region.label: region for region in regions}

        for instance in sample["annotations"]:
            category_id = instance["category_id"]

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
            }

            # Segmentation bitmap labels
            if (
                task_type == "segmentation-bitmap"
                or task_type == "segmentation-bitmap-highres"
            ):
                if instance["id"] not in regions:
                    # Only happens when the instance has 0 labeled pixels, which should not happen.
                    logger.warning(
                        f"Skipping instance with 0 labeled pixels: {sample['file_name']}, instance_id: {instance['id']}, category_id: {category_id}"
                    )
                    continue

                instance_mask = (
                    np.array(sample["segmentation_bitmap"], np.uint32) == instance["id"]
                )

                region = regions[instance["id"]]
                bbox = region.bbox
                # bbox = get_bbox(instance_mask)

                y0, x0, y1, x1 = bbox
                # rle = mask.encode(np.asfortranarray(instance_mask))
                rle = pctmask.encode(
                    np.array(instance_mask[:, :, None], dtype=np.uint8, order="F")
                )[
                    0
                ]  # https://github.com/matterport/Mask_RCNN/issues/387#issuecomment-522671380
                #         instance_mask_crop = instance_mask[y0:y1, x0:x1]
                #         rle = mask.encode(np.asfortranarray(instance_mask_crop))
                #         plt.imshow(instance_mask_crop)
                #         plt.show()

                # area = int(mask.area(rle))
                area = int(region.area)
                rle["counts"] = rle["counts"].decode("ascii")

                annotation.update(
                    {
                        "bbox": [x0, y0, x1 - x0, y1 - y0],
                        #             'bbox_mode': BoxMode.XYWH_ABS,
                        "segmentation": rle,
                        "area": area,
                        "iscrowd": 0,
                    }
                )

            # Vector labels
            elif "type" in instance:
                # bbox
                if instance["type"] == "bbox":
                    points = instance["points"]
                    x0 = points[0][0]
                    y0 = points[0][1]
                    x1 = points[1][0]
                    y1 = points[1][1]

                    annotation.update({"bbox": [x0, y0, x1 - x0, y1 - y0]})

                # keypoints
                elif instance["type"] == "point":
                    points = instance["points"]
                    x0 = points[0][0]
                    y0 = points[0][1]

                    annotation.update(
                        {
                            "keypoints": [
                                x0,
                                y0,
                                1,
                            ],  # https://cocodataset.org/#format-results
                        }
                    )

                # polygon
                elif instance["type"] == "polygon":
                    annotation.update({"points": instance["points"]})

                # polyline
                elif instance["type"] == "polyline":
                    logger.warning("Polyline annotations are not exported.")

                else:
                    assert False

            annotations.append(annotation)
            annotation_id += 1

    json_data = {
        "info": info,
        "categories": [category.dict() for category in categories],
        "images": images,
        "annotations": annotations
        # "segment_info": [] # Only in Panoptic annotations
    }

    file_name = os.path.join(
        export_folder,
        "export_coco-instance_{}_{}.json".format(
            dataset.dataset_identifier, dataset.release["name"]
        ),
    )
    with open(file_name, "w") as f:
        json.dump(json_data, f)

    print(f"Exported to {file_name}. Images in {dataset.image_dir}")
    return file_name, dataset.image_dir


def export_coco_panoptic(
    dataset: SegmentsDataset, export_folder: str
) -> Tuple[str, Optional[str]]:
    """Export a Segments dataset in COCO panoptic format.

    Args:
        dataset: A :class:`.SegmentsDataset`.
        export_folder: The output directory.
    Returns:
        Returns the file name and the image directory name.
    """

    # Create export folder
    os.makedirs(export_folder, exist_ok=True)

    # INFO
    info = {
        "description": dataset.release["dataset"]["name"],
        "version": dataset.release["name"],
        # 'year': '2022'
    }

    # CATEGORIES
    categories: List[SegmentsDatasetCategory] = []
    for i, category in enumerate(dataset.categories):
        color = category.color[:3] if category.color else get_color(i)
        isthing = (
            int(category.has_instances) if hasattr(category, "has_instances") else 0
        )

        categories.append(
            SegmentsDatasetCategory(
                id=category.id,
                name=category.name,
                color=color,
                isthing=isthing,
            )
        )

    categories_dict: Dict[int, SegmentsDatasetCategory] = {
        category.id: category for category in categories
    }
    id_generator = IdGenerator(categories_dict)

    # IMAGES AND ANNOTATIONS
    images = []
    annotations = []
    for i in tqdm(range(len(dataset)), total=len(dataset), colour="#FF9900"):
        sample = dataset[i]

        if sample["annotations"] is None:
            continue

        # Images
        image_id = i + 1
        images.append(
            {
                "id": image_id,
                "file_name": sample["file_name"],
                "height": sample["image"].size[1] if sample["image"] else None,
                "width": sample["image"].size[0] if sample["image"] else None,
            }
        )

        # Annotations
        panoptic_label = np.zeros(
            (
                sample["segmentation_bitmap"].size[1],
                sample["segmentation_bitmap"].size[0],
                3,
            ),
            np.uint8,
        )

        segments_info = []

        # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
        regions = regionprops(np.array(sample["segmentation_bitmap"], np.uint32))
        regions = {region.label: region for region in regions}

        for instance in sample["annotations"]:
            category_id = instance["category_id"]

            instance_id, color = id_generator.get_id_and_color(category_id)

            if instance["id"] not in regions:
                # Only happens when the instance has 0 labeled pixels, which should not happen.
                logger.warning(
                    f"Skipping instance with 0 labeled pixels: {sample['file_name']}, instance_id: {instance['id']}, category_id: {category_id}"
                )
                continue

            # Read the instance mask and fill in the panoptic label. TODO: take this out of the loop to speed things up.
            instance_mask = (
                np.array(sample["segmentation_bitmap"], np.uint32) == instance["id"]
            )
            panoptic_label[instance_mask] = color

            # bbox = get_bbox(instance_mask)
            region = regions[instance["id"]]
            bbox = region.bbox
            y0, x0, y1, x1 = bbox

            # rle = mask.encode(np.array(instance_mask[:,:,None], dtype=np.uint8, order='F'))[0] # https://github.com/matterport/Mask_RCNN/issues/387#issuecomment-522671380
            # area = int(mask.area(rle))
            area = int(region.area)

            segments_info.append(
                {
                    "id": instance_id,
                    "category_id": category_id,
                    "bbox": [x0, y0, x1 - x0, y1 - y0],
                    "area": area,
                    "iscrowd": 0,
                }
            )

        file_name = os.path.splitext(os.path.basename(sample["name"]))[0]
        label_file_name = f"{file_name}_label_{dataset.labelset}_coco-panoptic.png"
        annotations.append(
            {
                "segments_info": segments_info,
                "file_name": label_file_name,
                "image_id": image_id,
            }
        )

        # # Image
        # image = sample['image']
        # export_file = os.path.join(label_export_folder, '{}.png'.format(file_name))
        # image.save(export_file)

        # # Instance png
        # instance_label = sample['segmentation_bitmap']
        # export_file = os.path.join(dataset.image_dir, '{}_label_{}_instance.png'.format(file_name, dataset.labelset))
        # instance_label.save(export_file)

        # # Colored instance png
        # instance_label_colored = colorize(np.uint8(instance_label))
        # export_file = os.path.join(dataset.image_dir, '{}_label{}_instance_colored.png'.format(file_name, dataset.labelset))
        # Image.fromarray(img_as_ubyte(instance_label_colored)).save(export_file)

        # Panoptic png
        export_file = os.path.join(dataset.image_dir, label_file_name)
        Image.fromarray(panoptic_label).save(export_file)

        # # Semantic png
        # semantic_label = get_semantic_bitmap(instance_label, sample['annotations'])
        # export_file = os.path.join(dataset.image_dir, '{}_label_{}_semantic.png'.format(file_name, dataset.labelset))
        # Image.fromarray(img_as_ubyte(semantic_label)).save(export_file)

        # # Colored semantic png
        # semantic_label_colored = colorize(np.uint8(semantic_label), colormap=[c['color'] for c in categories])
        # export_file = os.path.join(dataset.image_dir, '{}_label_{}_semantic_colored.png'.format(file_name, dataset.labelset))
        # Image.fromarray(img_as_ubyte(semantic_label_colored)).save(export_file)

    # PUT EVERYTHING TOGETHER
    json_data = {
        "info": info,
        "categories": [category.dict() for category in categories],
        "images": images,
        "annotations": annotations,
    }

    # WRITE JSON TO FILE
    file_name = os.path.join(
        export_folder,
        "export_coco-panoptic_{}_{}.json".format(
            dataset.dataset_identifier, dataset.release["name"]
        ),
    )
    with open(file_name, "w") as f:
        json.dump(json_data, f)

    print(f"Exported to {file_name}. Images and labels in {dataset.image_dir}")
    return file_name, dataset.image_dir


def export_image(
    dataset: SegmentsDataset,
    export_folder: str,
    export_format: str,
    id_increment: int,
) -> Optional[str]:
    """Export a Segments dataset as images.

    Args:
        dataset: A :class:`.SegmentsDataset`.
        export_folder: The output directory.
        export_format: The export format.
        id_increment: Increment the category ids with this number.
    Returns:
        Returns the export folder.
    """
    # Create export folder

    if export_folder == ".":
        export_folder = dataset.image_dir

    os.makedirs(export_folder, exist_ok=True)

    # CATEGORIES
    categories = []
    for i, category in enumerate(dataset.categories):
        color = category.color[:3] if category.color else get_color(i)
        isthing = (
            int(category.has_instances) if hasattr(category, "has_instances") else 0
        )

        categories.append(
            SegmentsDatasetCategory.parse_obj(
                {
                    "id": category.id,
                    "name": category.name,
                    "color": color,
                    "isthing": isthing,
                }
            )
        )

    for i in tqdm(range(len(dataset)), total=len(dataset), colour="#FF9900"):
        sample = dataset[i]

        if sample["annotations"] is None:
            continue

        file_name = os.path.splitext(os.path.basename(sample["name"]))[0]

        # # Image
        # image = sample['image']
        # export_file = os.path.join(label_export_folder, '{}.png'.format(file_name))
        # image.save(export_file)

        if export_format == "instance":
            # Instance png
            instance_label = sample["segmentation_bitmap"]
            export_file = os.path.join(
                export_folder,
                f"{file_name}_label_{dataset.labelset}_instance.png",
            )
            instance_label.save(export_file)

        elif export_format == "instance-color":
            # Colored instance png
            instance_label = sample["segmentation_bitmap"]
            instance_label_colored = colorize(np.uint8(instance_label))
            export_file = os.path.join(
                export_folder,
                f"{file_name}_label_{dataset.labelset}_instance_colored.png",
            )
            Image.fromarray(img_as_ubyte(instance_label_colored)).save(export_file)

        elif export_format == "semantic":
            # Semantic png
            instance_label = sample["segmentation_bitmap"]
            semantic_label = get_semantic_bitmap(
                instance_label, sample["annotations"], id_increment
            )
            export_file = os.path.join(
                export_folder,
                f"{file_name}_label_{dataset.labelset}_semantic.png",
            )
            Image.fromarray(img_as_ubyte(semantic_label)).save(export_file)

        elif export_format == "semantic-color":
            # Colored semantic png
            instance_label = sample["segmentation_bitmap"]
            semantic_label = get_semantic_bitmap(
                instance_label, sample["annotations"], id_increment
            )
            semantic_label_colored = colorize(
                np.uint8(semantic_label), colormap=[c.color for c in categories]
            )
            export_file = os.path.join(
                export_folder,
                f"{file_name}_label_{dataset.labelset}_semantic_colored.png",
            )
            Image.fromarray(img_as_ubyte(semantic_label_colored)).save(export_file)

    print(f"Exported to {export_folder}")
    return export_folder


def export_yolo(
    dataset: SegmentsDataset,
    export_folder: str,
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
) -> Optional[str]:
    """Export a Segments dataset in YOLO format.

    Args:
        dataset: A segments dataset.
        export_folder: The output directory.
        image_width: The width of the image (needed for ``image-vector-sequence``).
        image_height: The height of the image (needed for ``image-vector-sequence``).
    Returns:
        The image directory of the dataset.
    Raises:
        :exc:`ValueError`: If the dataset is not a bounding box dataset.
        :exc:`ValueError`: If the dataset is an ``image-vector-sequence``and the image width or image height is :obj:`None`.
    """

    def write_yolo_file(
        file_name: str, annotations: Any, image_width: float, image_height: float
    ) -> None:
        with open(file_name, "w") as f:
            for annotation in annotations:
                if annotation["type"] == "bbox":
                    category_id = annotation["category_id"]
                    [[x0, y0], [x1, y1]] = annotation["points"]

                    # Normalize
                    x0, x1 = x0 / image_width, x1 / image_width
                    y0, y1 = y0 / image_height, y1 / image_height

                    # Get center, width and height of bbox
                    x_center = (x0 + x1) / 2
                    y_center = (y0 + y1) / 2
                    width = abs(x1 - x0)
                    height = abs(y1 - y0)

                    # Save it to the file
                    # print(category_id, x_center, y_center, width, height)
                    f.write(
                        "{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                            category_id, x_center, y_center, width, height
                        )
                    )

    # Create export folder
    os.makedirs(os.path.join(export_folder, dataset.image_dir), exist_ok=True)

    if dataset.task_type not in ["vector", "bboxes", "image-vector-sequence"]:
        raise ValueError("You can only export bounding box datasets to YOLO format.")

    if dataset.task_type == "vector":
        logger.warning(
            "Only bounding box annotations will be processed. Polygon, polyline and keypoint annotations will be ignored."
        )

    if dataset.task_type == "image-vector-sequence":
        logger.warning(
            "Note that the sequences will be exported as individual frames, disregarding the tracking information."
        )
        for i in tqdm(range(len(dataset)), total=len(dataset), colour="#FF9900"):
            sample = dataset[i]
            image_name = os.path.splitext(os.path.basename(sample["name"]))[0]

            if image_width is None or image_height is None:
                raise ValueError(
                    "Please provide image_width and image_height parameters (i.e., not None)."
                )

            for j, frame in enumerate(
                sample["labels"]["ground-truth"]["attributes"]["frames"]
            ):
                # Construct the file name from image and frame name
                try:
                    frame_name = sample["attributes"]["frames"][j]["name"]
                except (KeyError, TypeError):
                    frame_name = f"{j + 1:05d}"
                file_name = os.path.join(
                    export_folder, dataset.image_dir, f"{image_name}-{frame_name}.txt"
                )

                # Testing on x is the same as testing len(x)>0 (this also checks that x is not None - see truthy and falsy values in Python)
                # https://stackoverflow.com/questions/39983695/what-is-truthy-and-falsy-how-is-it-different-from-true-and-false
                if "annotations" in frame and frame["annotations"]:
                    annotations = frame["annotations"]
                    write_yolo_file(file_name, annotations, image_width, image_height)
    else:
        for i in tqdm(range(len(dataset)), total=len(dataset), colour="#FF9900"):
            sample = dataset[i]
            image_name = os.path.splitext(os.path.basename(sample["name"]))[0]
            file_name = os.path.join(
                export_folder, dataset.image_dir, f"{image_name}.txt"
            )

            if image_width is None or image_height is None:
                image_width = sample["image"].width
                image_height = sample["image"].height

            if "annotations" in sample and sample["annotations"]:
                annotations = sample["annotations"]
                write_yolo_file(
                    file_name,
                    annotations,
                    cast(float, image_width),
                    cast(float, image_height),
                )

    print(f"Exported. Images and labels in {dataset.image_dir}")
    return dataset.image_dir


def export_polygon(
    dataset: SegmentsDataset, export_folder: str
) -> Tuple[str, Optional[str]]:
    """Export a Segments dataset as polygons (i.e., contours).

    Args:
        dataset: A :class:`.SegmentsDataset`.
        export_folder: The output directory.
    Raises:
        :exc:`ImportError`: If OpenCV is not installed.
    Returns:
        Returns the file name and the image directory name.
    """
    try:
        import cv2 as cv
    except ImportError as e:
        logger.error("Please install OpenCV first: pip install opencv-python.")
        raise e

    # Create export folder
    os.makedirs(export_folder, exist_ok=True)

    categories = dataset.categories
    # task_type = dataset.task_type

    images = []
    annotations = []

    annotation_id = 1
    for i in tqdm(range(len(dataset)), total=len(dataset), colour="#FF9900"):
        sample = dataset[i]

        if sample["annotations"] is None:
            continue

        image_id = i + 1
        images.append(
            {
                "id": image_id,
                "file_name": sample["file_name"],
                "height": sample["image"].size[1] if sample["image"] else None,
                "width": sample["image"].size[0] if sample["image"] else None,
            }
        )

        # https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
        regions = regionprops(np.array(sample["segmentation_bitmap"], np.uint32))
        regions = {region.label: region for region in regions}

        for instance in sample["annotations"]:
            category_id = instance["category_id"]

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "attributes": instance.get("attributes", None),
            }

            if instance["id"] not in regions:
                # Only happens when the instance has 0 labeled pixels, which should not happen.
                logger.warning(
                    f"Skipping instance with 0 labeled pixels: {sample['file_name']}, instance_id: {instance['id']}, category_id: {category_id}"
                )
                continue

            instance_mask = (
                np.asarray(sample["segmentation_bitmap"], np.uint8) == instance["id"]
            )

            # Allowed OpenCV data types: https://stackoverflow.com/questions/12785121/access-opencv-matrix-cv-32s-element
            black_white_image = instance_mask.astype("uint8")

            # Contour retrieval modes (all, outer, in a hierarchy, etc.): https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
            contours, hierarchy = cv.findContours(
                black_white_image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE
            )

            # Make object json serializable
            serializable_contours = [
                np.squeeze(contour).tolist() for contour in contours
            ]

            annotation.update({"polygons": serializable_contours})

            annotations.append(annotation)
            annotation_id += 1

    json_data = {
        "categories": [category.dict() for category in categories],
        "images": images,
        "annotations": annotations,
    }

    file_name = os.path.join(
        export_folder,
        f"export_polygon_{dataset.dataset_identifier}_{dataset.release['name']}.json",
    )
    with open(file_name, "w") as f:
        json.dump(json_data, f)

    print(f"Exported to {file_name}. Images in {dataset.image_dir}")
    return file_name, dataset.image_dir
