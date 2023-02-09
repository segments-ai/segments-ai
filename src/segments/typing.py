from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel as PydanticBaseModel
from pydantic import validator
from segments.exceptions import ValidationError
from typing_extensions import Literal, TypedDict, get_args


class BaseModel(PydanticBaseModel):
    class Config:
        # Smart union checks all types before deciding which one to pick (otherwise, first type that fits). https://pydantic-docs.helpmanual.io/usage/model_config/#smart-union
        smart_union = True
        # What happens with extra fields in dictionaries. Use ignore in production and allow in debug mode. https://pydantic-docs.helpmanual.io/usage/model_config/#change-behaviour-globally
        extra = "ignore"
        # What happens with wrong field types. Use false in production and true in debug mode. https://pydantic-docs.helpmanual.io/usage/types/#arbitrary-types-allowed
        arbitrary_types_allowed = False


#######################################
# Literals, constants and other types #
#######################################
LabelStatus = Literal[
    "REVIEWED",
    "REVIEWING_IN_PROGRESS",
    "LABELED",
    "LABELING_IN_PROGRESS",
    "REJECTED",
    "PRELABELED",
    "SKIPPED",
    "UNLABELED",
    "VERIFIED",
]
TaskType = Literal[
    "segmentation-bitmap",
    "segmentation-bitmap-highres",
    "image-vector-sequence",
    "bboxes",
    "vector",
    "pointcloud-cuboid",
    "pointcloud-cuboid-sequence",
    "pointcloud-segmentation",
    "pointcloud-segmentation-sequence",
    "pointcloud-vector",
    "pointcloud-vector-sequence",
    "text-named-entities",
    "text-named-entities",
    "text-span-categorization",
    "",
]
Role = Literal["labeler", "reviewer", "admin"]
IssueStatus = Literal["OPEN", "CLOSED"]
Category = Literal[
    "street_scenery",
    "garden",
    "agriculture",
    "satellite",
    "people",
    "medical",
    "fruit",
    "other",
]
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]
FormatVersion = Union[float, str]
ObjectAttributes = Dict[str, Optional[Union[str, bool]]]
ImageAttributes = Dict[str, Optional[Union[str, bool]]]


class AuthHeader(TypedDict):
    Authorization: str


###########
# Release #
###########
class URL(BaseModel):
    url: Optional[
        str
    ]  # TODO Remove optional (e.g., the backend does not return an URL when adding a release).


class Release(BaseModel):
    uuid: str
    name: str
    description: str
    release_type: Literal["JSON"]
    attributes: URL
    status: Literal["PENDING", "SUCCEEDED", "FAILED"]
    # status_info: str
    created_at: str
    samples_count: int


#########
# Issue #
#########
class IssueComment(BaseModel):
    created_at: str
    created_by: str
    text: str


class Issue(BaseModel):
    uuid: str
    description: str
    created_at: str
    updated_at: str
    created_by: str
    updated_by: str
    comments: List[IssueComment]
    status: IssueStatus
    sample_uuid: str
    sample_name: str


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


#########
# Label #
#########
class Annotation(BaseModel):
    id: int
    category_id: int
    attributes: Optional[ObjectAttributes]


# Image segmenation
class ImageSegmentationLabelAttributes(BaseModel):
    annotations: List[Annotation]
    segmentation_bitmap: URL
    image_attributes: Optional[ImageAttributes]
    format_version: Optional[FormatVersion]


# Image vector
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class ImageVectorAnnotation(BaseModel):
    id: int
    category_id: int
    points: List[List[float]]
    type: Literal["bbox", "polygon", "polyline", "point"]
    attributes: Optional[ObjectAttributes]


class ImageVectorLabelAttributes(BaseModel):
    annotations: List[ImageVectorAnnotation]
    format_version: Optional[FormatVersion]
    image_attributes: Optional[ImageAttributes]


# Image sequence vector
class ImageSequenceVectorAnnotation(ImageVectorAnnotation):
    track_id: int
    is_keyframe: bool = False


class ImageVectorFrame(BaseModel):
    annotations: List[ImageSequenceVectorAnnotation]
    timestamp: Optional[str]
    format_version: Optional[FormatVersion]
    image_attributes: Optional[ImageAttributes]


class ImageSequenceVectorLabelAttributes(BaseModel):
    frames: List[ImageVectorFrame]
    format_version: Optional[FormatVersion]


# Point cloud segmentation
class PointcloudSegmentationLabelAttributes(BaseModel):
    annotations: List[Annotation]
    point_annotations: List[int]
    format_version: Optional[FormatVersion]


class XYZ(BaseModel):
    x: float
    y: float
    z: float


# Point cloud cuboid
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class PointcloudCuboidAnnotation(BaseModel):
    id: int
    category_id: int
    position: XYZ
    dimensions: XYZ
    yaw: float
    type: Literal["cuboid", "cuboid-sync"]
    attributes: Optional[ObjectAttributes]


class PointcloudCuboidLabelAttributes(BaseModel):
    annotations: List[PointcloudCuboidAnnotation]
    format_version: Optional[FormatVersion]


# Point cloud vector
class PointcloudVectorAnnotation(BaseModel):
    id: int
    category_id: int
    points: List[List[float]]
    type: Literal["polygon", "polyline", "point"]
    attributes: Optional[ObjectAttributes]


class PointcloudVectorLabelAttributes(BaseModel):
    annotations: List[PointcloudVectorAnnotation]
    format_version: Optional[FormatVersion]


# Point cloud sequence segmentation
class PointcloudSequenceSegmentationAnnotation(BaseModel):
    id: int
    category_id: int
    track_id: int
    is_keyframe: bool = False
    attributes: Optional[ObjectAttributes]


class PointcloudSegmentationFrame(BaseModel):
    annotations: List[PointcloudSequenceSegmentationAnnotation]
    point_annotations: Optional[List[int]]
    timestamp: Optional[str]
    format_version: Optional[FormatVersion]


class PointcloudSequenceSegmentationLabelAttributes(BaseModel):
    frames: List[PointcloudSegmentationFrame]
    format_version: Optional[FormatVersion]


# Point cloud sequence cuboid
class PointcloudSequenceCuboidAnnotation(PointcloudCuboidAnnotation):
    track_id: int
    is_keyframe: bool = False


class PointcloudSequenceCuboidFrame(BaseModel):
    annotations: List[PointcloudSequenceCuboidAnnotation]
    timestamp: Optional[str]
    format_version: Optional[FormatVersion]


class PointcloudSequenceCuboidLabelAttributes(BaseModel):
    frames: List[PointcloudSequenceCuboidFrame]
    format_version: Optional[FormatVersion]


# Point cloud sequence vector
class PointcloudSequenceVectorAnnotation(PointcloudVectorAnnotation):
    track_id: int
    is_keyframe: bool = False


class PointcloudSequenceVectorFrame(BaseModel):
    annotations: List[PointcloudSequenceVectorAnnotation]
    format_version: Optional[FormatVersion]
    timestamp: Optional[str]


class PointcloudSequenceVectorLabelAttributes(BaseModel):
    frames: List[PointcloudSequenceVectorFrame]
    format_version: Optional[FormatVersion]


# Text
class TextAnnotation(BaseModel):
    start: int
    end: int
    category_id: int


class TextLabelAttributes(BaseModel):
    annotations: List[TextAnnotation]
    format_version: Optional[FormatVersion]


# Most specific type first
# https://pydantic-docs.helpmanual.io/usage/types/#unions
LabelAttributes = Union[
    ImageVectorLabelAttributes,
    ImageSegmentationLabelAttributes,
    ImageSequenceVectorLabelAttributes,
    PointcloudCuboidLabelAttributes,
    PointcloudVectorLabelAttributes,
    PointcloudSegmentationLabelAttributes,
    PointcloudSequenceCuboidLabelAttributes,
    PointcloudSequenceVectorLabelAttributes,
    PointcloudSequenceSegmentationLabelAttributes,
    TextLabelAttributes,
]


class Label(BaseModel):
    sample_uuid: str
    label_type: TaskType
    label_status: LabelStatus
    labelset: str
    attributes: LabelAttributes
    created_at: str
    created_by: str
    updated_at: str
    score: Optional[float]
    rating: Optional[float]
    reviewed_at: Optional[str]
    reviewed_by: Optional[str]


##########
# Sample #
##########
# Image
class ImageSampleAttributes(BaseModel):
    image: URL


# Image sequence
class ImageFrame(ImageSampleAttributes):
    name: Optional[str]


class ImageSequenceSampleAttributes(BaseModel):
    frames: List[ImageFrame]


# Point cloud
class PCD(BaseModel):
    url: str
    type: Literal["pcd", "kitti", "nuscenes"] = "pcd"


class XYZW(BaseModel):
    qx: float
    qy: float
    qz: float
    qw: float


class EgoPose(BaseModel):
    position: XYZ
    heading: XYZW


class CameraIntrinsics(BaseModel):
    intrinsic_matrix: List[List[float]]


class CameraExtrinsics(BaseModel):
    translation: XYZ
    rotation: XYZW


class CalibratedImage(URL):
    row: int
    col: int
    intrinsics: Optional[CameraIntrinsics]
    extrinsics: Optional[CameraExtrinsics]


class PointcloudSampleAttributes(BaseModel):
    pcd: PCD
    images: Optional[List[CalibratedImage]]
    ego_pose: Optional[EgoPose]
    default_z: Optional[float]
    name: Optional[str]
    timestamp: Optional[str]


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
    assigned_labeler: Optional[str]
    assigned_reviewer: Optional[str]
    comments: Optional[List[str]]
    priority: float
    has_embedding: Optional[bool]
    label: Optional[Label]
    issues: Optional[List[Issue]]
    dataset_full_name: Optional[str]


########################
# Dataset and labelset #
########################
class User(BaseModel):
    username: str
    created_at: str


class Collaborator(BaseModel):
    user: User
    role: Role


class SelectTaskAttribute(BaseModel):
    name: str
    input_type: Literal["select"]
    values: List[str]
    default_value: Optional[str]


class TextTaskAttribute(BaseModel):
    name: str
    input_type: Literal["text"]
    default_value: Optional[str]


class NumberTaskAttribute(BaseModel):
    name: str
    input_type: Literal["number"]
    default_value: Optional[float]
    min: Optional[float]
    max: Optional[float]
    step: Optional[float]


class CheckboxTaskAttribute(BaseModel):
    name: str
    input_type: Literal["checkbox"]
    default_value: Optional[bool]


TaskAttribute = Union[
    SelectTaskAttribute,
    TextTaskAttribute,
    NumberTaskAttribute,
    CheckboxTaskAttribute,
]


class TaskAttributeCategory(BaseModel):
    name: str
    id: int
    color: Optional[Union[RGB, RGBA]]
    has_instances: Optional[bool]
    attributes: Optional[List[TaskAttribute]]
    dimensions: Optional[XYZ]

    class Config:
        extra = "allow"


class TaskAttributes(BaseModel):
    format_version: Optional[FormatVersion]
    categories: Optional[List[TaskAttributeCategory]]
    image_attributes: Optional[List[TaskAttribute]]

    class Config:
        extra = "allow"


class Owner(BaseModel):
    username: str
    created_at: str
    email: Optional[str]


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
    # uuid: Optional[str]
    # readme: Optional[str]
    # task_type: Optional[TaskType]
    # attributes: Optional[Union[str, TaskAttributes]]
    is_groundtruth: Optional[bool]
    # statistics: Optional[Statistics]
    created_at: Optional[str]
    # stats: Optional[Dict[str, Any]]


class LabelStats(BaseModel):
    TOTAL: Optional[int]
    LABELED: Optional[int]
    UNLABELED: Optional[int]
    PRELABELED: Optional[int]


class Dataset(BaseModel):
    name: str
    full_name: str
    cloned_from: Optional[str]
    description: str
    # data_type: Literal["IMAGE"]
    category: str  # Category
    public: bool
    owner: Owner
    created_at: str
    enable_ratings: bool
    enable_skip_labeling: bool
    enable_skip_reviewing: bool
    enable_save_button: bool
    enable_label_status_verified: bool
    enable_same_dimensions_track_constraint: bool
    enable_interpolation: bool
    task_type: TaskType
    # task_readme: str
    label_stats: LabelStats
    samples_count: Optional[Union[str, int]]
    collaborators_count: Optional[int]
    task_attributes: Optional[TaskAttributes]
    labelsets: Optional[List[Labelset]]
    role: Optional[str]
    readme: Optional[str]
    noncollaborator_can_label: Optional[bool]
    noncollaborator_can_review: Optional[bool]
    # tasks: Optional[List[Dict[str, Any]]]
    embeddings_enabled: Optional[bool]

    @validator("category")
    def check_category(cls, category: str) -> str:
        category_list = get_args(Category)
        if category not in category_list and "custom-" not in category:
            raise ValidationError(
                f"The category should be one of {category_list}, but is {category}."
            )
        return category


####################
# Segments dataset #
####################
class SegmentsDatasetCategory(BaseModel):
    id: int
    name: str
    color: Optional[Union[RGB, RGBA]]
    attributes: Optional[List[Any]]

    class Config:
        allow_population_by_field_name = True
        extra = "allow"
