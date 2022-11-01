from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, get_args

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Extra, validator
from segments.exceptions import ValidationError
from typing_extensions import Literal, TypedDict


class BaseModel(PydanticBaseModel):
    class Config:
        # https://pydantic-docs.helpmanual.io/usage/model_config/#smart-union
        smart_union = True


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
DataType = Literal["IMAGE"]
Role = Literal["labeler", "reviewer", "admin"]
Status = Literal["PENDING", "SUCCEEDED", "FAILED"]
IssueStatus = Literal["OPEN", "CLOSED"]
ReleaseType = Literal["JSON"]
ImageVectorAnnotationType = Literal["bbox", "polygon", "polyline", "point"]
PointcloudVectorAnnotationType = Literal["polygon", "polyline", "point"]
PCDType = Literal["pcd", "kitti", "nuscenes"]
InputType = Literal["select", "text", "number", "checkbox"]
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
    ] = None  # TODO Remove optional (e.g., the backend does not return an URL when adding a release).


class Release(BaseModel):
    uuid: str
    name: str
    description: str
    release_type: ReleaseType
    attributes: URL
    status: Status
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
# Helper classes
class Annotation(BaseModel):
    id: int
    category_id: int
    attributes: Optional[ObjectAttributes] = None


class SequenceFields(BaseModel):
    track_id: int
    is_keyframe: bool = False


class LabelAttributesFields(BaseModel):
    # https://mypy.readthedocs.io/en/stable/common_issues.html#invariance-vs-covariance
    annotations: Sequence[
        Annotation
    ]  # overwrite in subclass with specific annotation type
    image_attributes: Optional[ImageAttributes] = None
    format_version: Optional[FormatVersion] = None


class FrameFields(LabelAttributesFields):
    timestamp: Optional[int] = None


# Image segmenation
class ImageSegmentationLabelAttributes(LabelAttributesFields):
    segmentation_bitmap: URL


# Image vector
class ImageVectorAnnotation(Annotation):
    points: List[List[float]]
    type: ImageVectorAnnotationType


class ImageVectorLabelAttributes(LabelAttributesFields):
    annotations: List[ImageVectorAnnotation]


# Image sequence vector
class ImageSequenceVectorAnnotation(ImageVectorAnnotation, SequenceFields):
    pass


class ImageVectorFrame(FrameFields):
    annotations: List[ImageSequenceVectorAnnotation]


class ImageSequenceVectorLabelAttributes(BaseModel):
    frames: List[ImageVectorFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud segmentation
class PointcloudSegmentationLabelAttributes(LabelAttributesFields):
    point_annotations: List[int]


class XYZ(BaseModel):
    x: float
    y: float
    z: float


# Point cloud cuboid
class PointcloudCuboidAnnotation(Annotation):
    position: XYZ
    dimensions: XYZ
    yaw: float
    type: Literal["cuboid"]


class PointcloudCuboidLabelAttributes(LabelAttributesFields):
    annotations: List[PointcloudCuboidAnnotation]


# Point cloud vector
class PointcloudVectorAnnotation(Annotation):
    points: List[List[float]]
    type: PointcloudVectorAnnotationType


class PointcloudVectorLabelAttributes(LabelAttributesFields):
    annotations: List[PointcloudVectorAnnotation]


# Point cloud sequence segmentation
class PointcloudSequenceSegmentationAnnotation(Annotation, SequenceFields):
    pass


class PointcloudSegmentationFrame(FrameFields):
    annotations: List[PointcloudSequenceSegmentationAnnotation]
    point_annotations: Optional[List[int]] = None


class PointcloudSequenceSegmentationLabelAttributes(BaseModel):
    frames: List[PointcloudSegmentationFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud sequence cuboid
class PointcloudSequenceCuboidAnnotation(PointcloudCuboidAnnotation, SequenceFields):
    pass


class PointcloudSequenceCuboidFrame(FrameFields):
    annotations: List[PointcloudSequenceCuboidAnnotation]


class PointcloudSequenceCuboidLabelAttributes(BaseModel):
    frames: List[PointcloudSequenceCuboidFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud sequence vector
class PointcloudSequenceVectorAnnotation(PointcloudVectorAnnotation, SequenceFields):
    pass


class PointcloudSequenceVectorFrame(FrameFields):
    annotations: List[PointcloudSequenceVectorAnnotation]


class PointcloudSequenceVectorLabelAttributes(BaseModel):
    frames: List[PointcloudSequenceVectorFrame]
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
    PointcloudVectorLabelAttributes,
    PointcloudSequenceSegmentationLabelAttributes,
    PointcloudSequenceCuboidLabelAttributes,
    PointcloudSequenceVectorLabelAttributes,
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
    score: Optional[float] = None
    rating: Optional[float] = None
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None


##########
# Sample #
##########
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


class CameraIntrinsics(BaseModel):
    intrinsic_matrix: List[List[float]]


class CameraExtrinsics(BaseModel):
    translation: XYZ
    rotation: XYZW


class CalibratedImage(URL):
    row: int
    col: int
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None


class PointcloudSampleAttributes(BaseModel):
    pcd: PCD
    images: Optional[List[CalibratedImage]] = None
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
    default_value: Optional[str] = None


class TextTaskAttribute(BaseModel):
    name: str
    input_type: Literal["text"]
    default_value: Optional[str] = None


class NumberTaskAttribute(BaseModel):
    name: str
    input_type: Literal["number"]
    default_value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None


class CheckboxTaskAttribute(BaseModel):
    name: str
    input_type: Literal["checkbox"]
    default_value: Optional[bool] = None


TaskAttribute = Union[
    SelectTaskAttribute,
    TextTaskAttribute,
    NumberTaskAttribute,
    CheckboxTaskAttribute,
]


class TaskAttributeCategory(BaseModel):
    name: str
    id: int
    color: Optional[Union[RGB, RGBA]] = None
    attributes: Optional[List[TaskAttribute]] = None
    dimensions: Optional[XYZ] = None


class TaskAttributes(BaseModel):
    format_version: Optional[FormatVersion] = None
    categories: Optional[List[TaskAttributeCategory]] = None

    class Config:
        extra = Extra.allow


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
    # uuid: Optional[str] = None
    # readme: Optional[str] = None
    # task_type: Optional[TaskType] = None
    # attributes: Optional[Union[str, TaskAttributes]] = None
    is_groundtruth: Optional[bool] = None
    # statistics: Optional[Statistics] = None
    created_at: Optional[str] = None
    # stats: Optional[Dict[str, Any]] = None


class LabelStats(BaseModel):
    TOTAL: Optional[int] = None
    LABELED: Optional[int] = None
    UNLABELED: Optional[int] = None
    PRELABELED: Optional[int] = None


class Dataset(BaseModel):
    name: str
    full_name: str
    cloned_from: Optional[str] = None
    description: str
    # data_type: DataType
    category: str  # Category
    public: bool
    owner: Owner
    created_at: str
    enable_ratings: bool
    enable_skip_labeling: bool
    enable_skip_reviewing: bool
    enable_save_button: bool
    enable_same_dimensions_track_constraint: bool
    enable_interpolation: bool
    task_type: TaskType
    # task_readme: str
    label_stats: LabelStats
    samples_count: Optional[Union[str, int]] = None
    collaborators_count: Optional[int] = None
    task_attributes: Optional[TaskAttributes] = None
    labelsets: Optional[List[Labelset]] = None
    role: Optional[str] = None
    readme: Optional[str] = None
    noncollaborator_can_label: Optional[bool] = None
    noncollaborator_can_review: Optional[bool] = None
    # tasks: Optional[List[Dict[str, Any]]] = None
    embeddings_enabled: Optional[bool] = None

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
    color: Optional[Union[RGB, RGBA]] = None
    attributes: Optional[List[Any]] = None

    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow
