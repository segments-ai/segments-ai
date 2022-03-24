from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict, Literal
from pydantic import BaseModel

####################################
# Enums, constants and other types #
####################################
# class LabelStatus(str, Enum):
#     reviewed = "REVIEWED"
#     reviewing_in_progress = "REVIEWING_IN_PROGRESS"
#     labeled = "LABELED"
#     labeling_in_progress = "LABELING_IN_PROGRESS"
#     rejected = "REJECTED"
#     prelabeled = "PRELABELED"
#     skipped = "SKIPPED"
LabelStatus = Literal[
    "REVIEWED",
    "REVIEWING_IN_PROGRESS",
    "LABELED",
    "LABELING_IN_PROGRESS",
    "REJECTED",
    "PRELABELED",
    "SKIPPED",
]


# class TaskType(str, Enum):
#     segmentation_bitmap = "segmentation-bitmap"
#     segmentation_bitmap_highres = "segmentation-bitmap-highres"
#     bboxes = "bboxes"
#     vector = "vector"
#     pointcloud_cuboid = "pointcloud-cuboid"
#     pointcloud_cuboid_sequence = "pointcloud-cuboid-sequence"
#     pointcloud_segmentation = "pointcloud-segmentation"
#     pointcloud_segmentation_sequence = "pointcloud-segmentation-sequence"
#     text_named_entities = "text-named-entities"
#     text_span_categorization = "text-span-categorization"
#     image_vector_sequence = "image-vector-sequence"
#     other = ""
TaskType = Literal[
    "segmentation-bitmap",
    "segmentation-bitmap-highres",
    "bboxes",
    "vector",
    "pointcloud-cuboid",
    "pointcloud-cuboid-sequence",
    "pointcloud-segmentation",
    "pointcloud-segmentation-sequence",
    "text-named-entities",
    "text-span-categorization",
    "image-vector-sequence",
    "",
]


# class DataType(str, Enum):
#     image = "IMAGE"
DataType = Literal["IMAGE"]


# class Role(str, Enum):
#     labeler = "labeler"
#     reviewer = "reviewer"
#     admin = "admin"
Role = Literal["labeler", "reviewer", "admin"]


# class Status(str, Enum):
#     pending = "PENDING"
#     succeeded = "SUCCEEDED"
#     failed = "FAILED"
Status = Literal["PENDING", "SUCCEEDED", "FAILED"]


# class ReleaseType(str, Enum):
#     json = "JSON"
ReleaseType = Literal["JSON"]


# class ImageVectorAnnotationType(str, Enum):
#     bbox = "bbox"
#     polygon = "polygon"
#     polyline = "polyline"
#     point = "point"
ImageVectorAnnotationType = Literal["bbox", "polygon", "polyline", "point"]


# class PointcloudAnnotationType(str, Enum):
#     cuboid = "cuboid"
PointcloudAnnotationType = Literal["cuboid"]

# class PCDType(str, Enum):
#     pcd = "pcd"
#     kitti = "kitti"
#     nuscenes = "nuscenes"
PCDType = Literal["pcd", "kitti", "nuscenes"]


# class InputType(str, Enum):
#     select = "select"
#     text = "text"
#     number = "number"
#     checkbox = "checkbox"
InputType = Literal["select", "text", "number", "checkbox"]


# class Category(str, Enum):
#     street_scenery = "street_scenery"
#     garden = "garden"
#     agriculture = "agriculture"
#     satellite = "satellite"
#     people = "people"
#     medical = "medical"
#     other = "other"
Category = Literal[
    "street_scenery", "garden", "agriculture", "satellite", "people", "medical", "other"
]

# DACITE_CONFIG = Config(
#     cast=[
#         LabelStatus,
#         TaskType,
#         DataType,
#         Role,
#         Status,
#         ReleaseType,
#         ImageVectorAnnotationType,
#         PointcloudAnnotationType,
#         PCDType,
#         InputType,
#         Category,
#     ]
# )
RGB = List[float]  # TODO Tuple[float, float, float]
RGBA = List[float]  # TODO Tuple[float, float, float, float]
FormatVersion = Union[float, str]


class AuthHeader(TypedDict):
    Authorization: str


###########
# Release #
###########
class URL(BaseModel):
    url: Optional[str] = None


class Release(BaseModel):
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


class PresignedPostFields(BaseModel):
    url: str
    fields: AWSFields


class File(BaseModel):
    uuid: str
    filename: str
    url: str
    created_at: str
    presignedPostFields: PresignedPostFields


#####################################
# Object and image level attributes #
#####################################
class ObjectAttribute(BaseModel):
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
            assert self.input_type == "select"  # InputType.select
        # Text/select
        if isinstance(self.default_value, str):
            assert (
                self.input_type == "text"
                or self.input_type
                == "select"  # InputType.text or self.input_type == InputType.select
            )
        # Number
        if (
            isinstance(self.min, float)
            or isinstance(self.max, float)
            or isinstance(self.step, float)
            or isinstance(self.default_value, float)
        ):
            assert self.input_type == "number"  # InputType.number
        # Checkbox
        if isinstance(self.default_value, bool):
            assert self.input_type == "checkbox"  # InputType.checkbox


ObjectAttributes = List[ObjectAttribute]
ImageAttributes = Dict[str, str]

# #########
# # Label #
# #########
class Annotation(BaseModel):
    id: int
    category_id: int
    attributes: Optional[ObjectAttributes] = None


# Image segmenation
class ImageSegmentationLabelAttributes(BaseModel):
    annotations: List[Annotation]
    segmentation_bitmap: URL
    image_attributes: Optional[ImageAttributes] = None
    format_version: Optional[FormatVersion] = None


# Image vector
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class _ImageVectorAnnotationBase(BaseModel):
    id: int
    category_id: int
    points: List[List[float]]
    type: ImageVectorAnnotationType


class ImageVectorAnnotation(_ImageVectorAnnotationBase):
    attributes: Optional[ObjectAttributes] = None


class ImageVectorLabelAttributes(BaseModel):
    annotations: List[ImageVectorAnnotation]
    format_version: Optional[FormatVersion] = None
    image_attributes: Optional[ImageAttributes] = None


# Image sequence vector
class ImageSequenceVectorAnnotation(_ImageVectorAnnotationBase):
    track_id: int
    is_keyframe: bool = False
    attributes: Optional[ObjectAttributes] = None


class ImageVectorFrame(BaseModel):
    annotations: List[ImageSequenceVectorAnnotation]
    timestamp: Optional[int] = None
    format_version: Optional[FormatVersion] = None
    image_attributes: Optional[ImageAttributes] = None


class ImageSequenceVectorLabelAttributes(BaseModel):
    frames: List[ImageVectorFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud segmentation
class PointcloudSegmentationLabelAttributes(BaseModel):
    annotations: List[Annotation]
    point_annotations: List[int]
    format_version: Optional[FormatVersion] = None


class XYZ(BaseModel):
    x: float
    y: float
    z: float


# Point cloud cuboid
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class _PointcloudCuboidAnnotationBase(BaseModel):
    id: int
    category_id: int
    position: XYZ
    dimensions: XYZ
    yaw: float
    type: PointcloudAnnotationType


class PointcloudCuboidAnnotation(_PointcloudCuboidAnnotationBase):
    attributes: Optional[ObjectAttributes] = None


class PointcloudCuboidLabelAttributes(BaseModel):
    annotations: List[PointcloudCuboidAnnotation]
    format_version: Optional[FormatVersion] = None


# Point cloud sequence segmentation
class PointcloudSequenceSegmentationAnnotation(BaseModel):
    id: int
    category_id: int
    track_id: int
    is_keyframe: bool = False
    attributes: Optional[ObjectAttributes] = None


class PointcloudSegmentationFrame(BaseModel):
    annotations: List[PointcloudSequenceSegmentationAnnotation]
    point_annotations: Optional[List[int]] = None
    format_version: Optional[FormatVersion] = None


class PointcloudSequenceSegmentationLabelAttributes(BaseModel):
    frames: List[PointcloudSegmentationFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud sequence cuboid
class PointcloudSequenceCuboidAnnotation(_PointcloudCuboidAnnotationBase):
    track_id: int
    is_keyframe: bool = False
    attributes: Optional[ObjectAttributes] = None


class PointcloudSequenceCuboidFrame(BaseModel):
    timestamp: int
    annotations: List[PointcloudSequenceCuboidAnnotation]
    format_version: Optional[FormatVersion] = None


class PointcloudSequenceCuboidLabelAttributes(BaseModel):
    frames: List[PointcloudSequenceCuboidFrame]
    format_version: Optional[FormatVersion] = None


# Text
class TextAnnotation(BaseModel):
    start: int
    end: int
    category_id: int


class TextLabelAttributes(BaseModel):
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


class Label(BaseModel):
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
class ImageSampleAttributes(BaseModel):
    image: URL


# Image sequence
class ImageFrame(ImageSampleAttributes):
    name: Optional[str] = None


class ImageSequenceSampleAttributes(BaseModel):
    frames: List[ImageFrame]


# Point cloud
class PCD(BaseModel):
    url: str
    type: PCDType = "pcd"


class XYZW(BaseModel):
    qx: float
    qy: float
    qz: float
    qw: float


class EgoPose(BaseModel):
    position: XYZ
    heading: XYZW


class PointcloudSampleAttributes(BaseModel):
    pcd: PCD
    ego_pose: Optional[EgoPose] = None
    default_z: Optional[float] = None
    name: Optional[str] = None
    timestamp: Optional[int] = None


# Point cloud sequence
class PointcloudSequenceSampleAttributes(BaseModel):
    frames: List[PointcloudSampleAttributes]


# Text
class TextSampleAttributes(BaseModel):
    text: str


SampleAttributes = Union[
    ImageSampleAttributes,
    ImageSequenceSampleAttributes,
    PointcloudSampleAttributes,
    PointcloudSequenceSampleAttributes,
    TextSampleAttributes,
]


class Sample(BaseModel):
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
class User(BaseModel):
    username: str
    created_at: str


class Collaborator(BaseModel):
    user: User
    role: Role


class TaskAttributeCategory(BaseModel):
    name: str
    id: int
    color: Optional[Union[RGB, RGBA]] = None
    attributes: Optional[ObjectAttributes] = None
    dimensions: Optional[XYZ] = None


class TaskAttributes(BaseModel):
    format_version: Optional[FormatVersion] = None
    categories: Optional[List[TaskAttributeCategory]] = None


class Owner(BaseModel):
    username: str
    created_at: str
    email: Optional[str] = None


class Statistics(BaseModel):
    prelabeled_count: int
    labeled_count: int
    reviewed_count: int
    rejected_count: int
    skipped_count: int
    samples_count: int


class Labelset(BaseModel):
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


class LabelStats(BaseModel):
    TOTAL: Optional[int] = None
    LABELED: Optional[int] = None
    UNLABELED: Optional[int] = None
    PRELABELED: Optional[int] = None


class Dataset(BaseModel):
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
