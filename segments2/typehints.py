from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict
from dacite import Config

####################################
# Enums, constants and other types #
####################################
class LabelStatus(Enum):
    reviewed = "REVIEWED"
    reviewing_in_progress = "REVIEWING_IN_PROGRESS"
    labeled = "LABELED"
    labeling_in_progress = "LABELING_IN_PROGRESS"
    rejected = "REJECTED"
    prelabeled = "PRELABELED"
    skipped = "SKIPPED"


class TaskType(Enum):
    segmentation_bitmap = "segmentation-bitmap"
    segmentation_bitmap_highres = "segmentation-bitmap-highres"
    bboxes = "bboxes"
    vector = "vector"
    pointcloud_cuboid = "pointcloud-cuboid"
    pointcloud_cuboid_sequence = "pointcloud-cuboid-sequence"
    pointcloud_segmentation = "pointcloud-segmentation"
    pointcloud_segmentation_sequence = "pointcloud-segmentation-sequence"
    text_named_entities = "text-named-entities"
    text_span_categorization = "text-span-categorization"
    image_vector_sequence = "image-vector-sequence"
    other = ""


class DataType(Enum):
    image = "IMAGE"


class Role(Enum):
    labeler = "labeler"
    reviewer = "reviewer"
    admin = "admin"


class Status(Enum):
    pending = "PENDING"
    succeeded = "SUCCEEDED"
    failed = "FAILED"


class ReleaseType(Enum):
    json = "JSON"


class ImageVectorAnnotationType(Enum):
    bbox = "bbox"
    polygon = "polygon"
    polyline = "polyline"
    point = "point"


class PointcloudAnnotationType(Enum):
    cuboid = "cuboid"


class PCDType(Enum):
    pcd = "pcd"
    kitti = "kitti"
    nuscenes = "nuscenes"


class InputType(Enum):
    select = "select"
    text = "text"
    number = "number"
    checkbox = "checkbox"


class Category(Enum):
    street_scenery = "street_scenery"
    garden = "garden"
    agriculture = "agriculture"
    satellite = "satellite"
    people = "people"
    medical = "medical"
    other = "other"


DACITE_CONFIG = Config(
    cast=[
        LabelStatus,
        TaskType,
        DataType,
        Role,
        Status,
        ReleaseType,
        ImageVectorAnnotationType,
        PointcloudAnnotationType,
        PCDType,
        InputType,
        Category,
    ]
)
RGB = List[float]  # TODO Tuple[float, float, float]
RGBA = List[float]  # TODO Tuple[float, float, float, float]
FormatVersion = Union[float, str]


class AuthHeader(TypedDict):
    Authorization: str


###########
# Release #
###########
@dataclass
class URL:
    url: Optional[str] = None


@dataclass
class Release:
    uuid: str
    name: str
    description: str
    release_type: ReleaseType
    attributes: URL
    status: Status
    status_info: str
    created_at: str
    samples_count: int


########
# File #
########
# https://stackoverflow.com/questions/60003444/typeddict-when-keys-have-invalid-names
AWSFields = TypedDict(
    "AWSFields",
    {
        "acl": str,
        "Content-Type": str,
        "key": str,
        "x-amz-algorithm": str,
        "x-amz-credential": str,
        "x-amz-date": str,
        "policy": str,
        "x-amz-signature": str,
    },
)


@dataclass
class PresignedPostFields:
    url: str
    fields: AWSFields  # Dict[str, Any]  # AWSFields


@dataclass
class File:
    uuid: str
    filename: str
    url: str
    created_at: str
    presignedPostFields: PresignedPostFields


#####################################
# Object and image level attributes #
#####################################
@dataclass
class ObjectAttribute:
    # Select and checkbox
    name: str
    input_type: InputType
    default_value: Optional[Union[str, float, bool]] = None
    # Text
    values: Optional[List[str]] = None
    # Number
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    def __post_init__(self):
        # Select
        if isinstance(self.values, list):
            assert self.input_type == InputType.select
        # Text/select
        if isinstance(self.default_value, str):
            assert (
                self.input_type == InputType.text or self.input_type == InputType.select
            )
        # Number
        if (
            isinstance(self.min, float)
            or isinstance(self.max, float)
            or isinstance(self.step, float)
            or isinstance(self.default_value, float)
        ):
            assert self.input_type == InputType.number
        # checkbox
        if isinstance(self.default_value, bool):
            assert self.input_type == InputType.checkbox


ObjectAttributes = List[ObjectAttribute]
ImageAttributes = Dict[str, str]

# #########
# # Label #
# #########
@dataclass
class Annotation:
    id: int
    category_id: int
    attributes: Optional[ObjectAttributes] = None


# Image segmenation
@dataclass
class ImageSegmentationLabelAttributes:
    annotations: List[Annotation]
    segmentation_bitmap: URL
    image_attributes: Optional[ImageAttributes] = None
    format_version: Optional[FormatVersion] = None


# Image vector
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
@dataclass
class _ImageVectorAnnotationBase:
    id: int
    category_id: int
    points: List[List[float]]
    type: ImageVectorAnnotationType


@dataclass
class ImageVectorAnnotation(_ImageVectorAnnotationBase):
    attributes: Optional[ObjectAttributes] = None


@dataclass
class ImageVectorLabelAttributes:
    annotations: List[ImageVectorAnnotation]
    format_version: Optional[FormatVersion] = None
    image_attributes: Optional[ImageAttributes] = None


# Image sequence vector
@dataclass
class ImageSequenceVectorAnnotation(_ImageVectorAnnotationBase):
    track_id: int
    is_keyframe: bool = False
    attributes: Optional[ObjectAttributes] = None


@dataclass
class ImageVectorFrame:
    annotations: List[ImageSequenceVectorAnnotation]
    timestamp: Optional[int] = None
    format_version: Optional[FormatVersion] = None
    image_attributes: Optional[ImageAttributes] = None


@dataclass
class ImageSequenceVectorLabelAttributes:
    frames: List[ImageVectorFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud segmentation
@dataclass
class PointcloudSegmentationLabelAttributes:
    annotations: List[Annotation]
    point_annotations: List[int]
    format_version: Optional[FormatVersion] = None


@dataclass
class XYZ:
    x: float
    y: float
    z: float


# Point cloud cuboid
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
@dataclass
class _PointcloudCuboidAnnotationBase:
    id: int
    category_id: int
    position: XYZ
    dimensions: XYZ
    yaw: float
    type: PointcloudAnnotationType


@dataclass
class PointcloudCuboidAnnotation(_PointcloudCuboidAnnotationBase):
    attributes: Optional[ObjectAttributes] = None


@dataclass
class PointcloudCuboidLabelAttributes:
    annotations: List[PointcloudCuboidAnnotation]
    format_version: Optional[FormatVersion] = None


# Point cloud sequence segmentation
@dataclass
class PointcloudSequenceSegmentationAnnotation:
    id: int
    category_id: int
    track_id: int
    is_keyframe: bool = False
    attributes: Optional[ObjectAttributes] = None


@dataclass
class PointcloudSegmentationFrame:
    annotations: List[PointcloudSequenceSegmentationAnnotation]
    point_annotations: Optional[List[int]] = None
    format_version: Optional[FormatVersion] = None


@dataclass
class PointcloudSequenceSegmentationLabelAttributes:
    frames: List[PointcloudSegmentationFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud sequence cuboid
@dataclass
class PointcloudSequenceCuboidAnnotation(_PointcloudCuboidAnnotationBase):
    track_id: int
    is_keyframe: bool = False
    attributes: Optional[ObjectAttributes] = None


@dataclass
class PointcloudSequenceCuboidFrame:
    timestamp: int
    annotations: List[PointcloudSequenceCuboidAnnotation]
    format_version: Optional[FormatVersion] = None


@dataclass
class PointcloudSequenceCuboidLabelAttributes:
    frames: List[PointcloudSequenceCuboidFrame]
    format_version: Optional[FormatVersion] = None


# Text
@dataclass
class TextAnnotation:
    start: int
    end: int
    category_id: int


@dataclass
class TextLabelAttributes:
    annotations: List[TextAnnotation]
    format_version: Optional[FormatVersion] = None


LabelAttributes = Union[
    ImageSegmentationLabelAttributes,
    ImageVectorLabelAttributes,
    ImageSequenceVectorLabelAttributes,
    PointcloudSegmentationLabelAttributes,
    PointcloudCuboidLabelAttributes,
    PointcloudSequenceSegmentationLabelAttributes,
    PointcloudSequenceCuboidLabelAttributes,
    TextLabelAttributes,
]


@dataclass
class Label:
    sample_uuid: str
    label_type: TaskType
    label_status: LabelStatus
    attributes: LabelAttributes
    created_at: str
    created_by: str
    updated_at: str
    score: Optional[float] = None
    rating: Optional[float] = None
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None


# ##########
# # Sample #
# ##########
# Image
@dataclass
class ImageSampleAttributes:
    image: URL


# Image sequence
@dataclass
class ImageFrame(ImageSampleAttributes):
    name: Optional[str] = None


@dataclass
class ImageSequenceSampleAttributes:
    frames: List[ImageFrame]


# Point cloud
@dataclass
class PCD:
    url: str
    type: PCDType = PCDType.pcd


@dataclass
class XYZW:
    qx: float
    qy: float
    qz: float
    qw: float


@dataclass
class EgoPose:
    position: XYZ
    heading: XYZW


@dataclass
class PointcloudSampleAttributes:
    pcd: PCD
    ego_pose: Optional[EgoPose] = None
    default_z: Optional[float] = None
    name: Optional[str] = None
    timestamp: Optional[int] = None


# Point cloud sequence
@dataclass
class PointcloudSequenceSampleAttributes:
    frames: List[PointcloudSampleAttributes]


# Text
@dataclass
class TextSampleAttributes:
    text: str


SampleAttributes = Union[
    ImageSampleAttributes,
    ImageSequenceSampleAttributes,
    PointcloudSampleAttributes,
    PointcloudSequenceSampleAttributes,
    TextSampleAttributes,
]


@dataclass
class Sample:
    uuid: str
    name: str
    attributes: SampleAttributes
    metadata: Dict[str, Any]
    created_at: str
    created_by: str
    comments: List[str]
    priority: float
    has_embedding: bool


########################
# Dataset and labelset #
########################
@dataclass
class User:
    username: str
    created_at: str


@dataclass
class Collaborator:
    user: User
    role: Role


@dataclass
class TaskAttributeCategory:
    name: str
    id: int
    color: Optional[Union[RGB, RGBA]] = None
    attributes: Optional[ObjectAttributes] = None
    dimensions: Optional[XYZ] = None


@dataclass
class TaskAttributes:
    format_version: Optional[FormatVersion] = None
    categories: Optional[List[TaskAttributeCategory]] = None


@dataclass
class Owner:
    username: str
    created_at: str
    email: Optional[str] = None


@dataclass
class Statistics:
    prelabeled_count: int
    labeled_count: int
    reviewed_count: int
    rejected_count: int
    skipped_count: int
    samples_count: int


@dataclass
class Labelset:
    name: str
    description: str
    uuid: Optional[str] = None
    readme: Optional[str] = None
    task_type: Optional[TaskType] = None
    attributes: Optional[Union[str, TaskAttributes]] = None
    is_groundtruth: Optional[bool] = None
    statistics: Optional[Statistics] = None
    created_at: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


@dataclass
class LabelStats:
    TOTAL: Optional[int] = None
    LABELED: Optional[int] = None
    UNLABELED: Optional[int] = None
    PRELABELED: Optional[int] = None


@dataclass
class Dataset:
    name: str
    description: str
    data_type: DataType
    category: Category
    public: bool
    owner: Owner
    created_at: str
    enable_ratings: bool
    enable_skip_labeling: bool
    enable_skip_reviewing: bool
    enable_save_button: bool
    task_type: TaskType
    task_readme: str
    label_stats: LabelStats
    samples_count: Optional[Union[str, int]] = None
    collaborators_count: Optional[int] = None
    cloned_from: Optional[int] = None
    task_attributes: Optional[TaskAttributes] = None
    labelsets: Optional[List[Labelset]] = None
    role: Optional[str] = None
    readme: Optional[str] = None
    noncollaborator_can_label: Optional[bool] = None
    noncollaborator_can_review: Optional[bool] = None
    tasks: Optional[List[Dict[str, Any]]] = None
    embeddings_enabled: Optional[bool] = None
