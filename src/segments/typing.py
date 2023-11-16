from __future__ import annotations

from enum import Enum as BaseEnum
from enum import EnumMeta as BaseEnumMeta
from typing import Any

from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, field_validator
from segments.exceptions import ValidationError
from typing_extensions import Literal, TypedDict, TypeAlias


class BaseModel(PydanticBaseModel):
    model_config = ConfigDict(
        # What happens with extra fields in dictionaries. Use ignore in production and allow in debug mode https://pydantic-docs.helpmanual.io/usage/model_config/#change-behaviour-globally
        extra="ignore",
        # What happens with wrong field types. Use false in production and true in debug mode https://pydantic-docs.helpmanual.io/usage/types/#arbitrary-types-allowed
        arbitrary_types_allowed=False,
        # Whether to populate models with the value property of enums, rather than the raw enum
        use_enum_values=True,
    )


class EnumMeta(BaseEnumMeta):
    # https://stackoverflow.com/questions/43634618/how-do-i-test-if-int-value-exists-in-python-enum-without-using-try-catch
    def __contains__(self, item):
        return isinstance(item, self) or item in {
            v.value for v in self.__members__.values()
        }

    # https://stackoverflow.com/questions/29503339/how-to-get-all-values-from-python-enum-class
    def __str__(self):
        return ", ".join(c.value for c in self)

    def __repr__(self):
        return self.__str__()


class Enum(BaseEnum, metaclass=EnumMeta):
    pass


#################################
# Literals, constants and enums #
#################################
class LabelStatus(str, Enum):
    REVIEWED = "REVIEWED"
    REVIEWING_IN_PROGRESS = "REVIEWING_IN_PROGRESS"
    LABELED = "LABELED"
    LABELING_IN_PROGRESS = "LABELING_IN_PROGRESS"
    REJECTED = "REJECTED"
    PRELABELED = "PRELABELED"
    SKIPPED = "SKIPPED"
    VERIFIED = "VERIFIED"
    UNLABELED = "UNLABELED"


# keep in sync with LabelStatus
class LabelStats(BaseModel):
    REVIEWED: int | None = None
    REVIEWING_IN_PROGRESS: int | None = None
    LABELED: int | None = None
    LABELING_IN_PROGRESS: int | None = None
    REJECTED: int | None = None
    PRELABELED: int | None = None
    SKIPPED: int | None = None
    VERIFIED: int | None = None
    UNLABELED: int | None = None
    # extra
    TOTAL: int | None = None


class TaskType(str, Enum):
    SEGMENTATION_BITMAP = "segmentation-bitmap"
    SEGMENTATION_BITMAP_HIGHRES = "segmentation-bitmap-highres"
    IMAGE_SEGMENTATION_SEQUENCE = "image-segmentation-sequence"
    BBOXES = "bboxes"
    VECTOR = "vector"
    IMAGE_VECTOR_SEQUENCE = "image-vector-sequence"
    KEYPOINTS = "keypoints"
    POINTCLOUD_CUBOID = "pointcloud-cuboid"
    POINTCLOUD_CUBOID_SEQUENCE = "pointcloud-cuboid-sequence"
    POINTCLOUD_SEGMENTATION = "pointcloud-segmentation"
    POINTCLOUD_SEGMENTATION_SEQUENCE = "pointcloud-segmentation-sequence"
    POINTCLOUD_VECTOR = "pointcloud-vector"
    POINTCLOUD_VECTOR_SEQUENCE = "pointcloud-vector-sequence"
    # MULTISENSOR = "multisensor"
    MULTISENSOR_SEQUENCE = "multisensor-sequence"  # combination of pointcloud-cuboid-sequence and image-vector-sequence
    TEXT_NAMED_ENTITIES = "text-named-entities"
    TEXT_SPAN_CATEGORIZATION = "text-span-categorization"
    EMPTY = ""


class Role(str, Enum):
    LABELER = "labeler"
    REVIEWER = "reviewer"
    MANAGER = "manager"
    ADMIN = "admin"


class IssueStatus(str, Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class Category(str, Enum):
    STREET_SCENERY = "street_scenery"
    GARDEN = "garden"
    AGRICULTURE = "agriculture"
    SATELLITE = "satellite"
    PEOPLE = "people"
    MEDICAL = "medical"
    FRUIT = "fruit"
    OTHER = "other"


class CameraConvention(str, Enum):
    OPEN_CV = "OpenCV"
    OPEN_GL = "OpenGL"


class InputType(str, Enum):
    SELECT = "select"
    TEXT = "text"
    NUMBER = "number"
    CHECKBOX = "checkbox"


class CameraDistortionModel(str, Enum):
    FISH_EYE = "fisheye"
    BROWN_CONRADY = "brown-conrady"


class PCDType(str, Enum):
    PCD = "pcd"
    BINARY_XYZI = "binary-xyzi"
    KITTI = "kitti"
    BINARY_XYZIR = "binary-xyzir"
    NUSCENES = "nuscenes"
    PLY = "ply"


class ReleaseStatus(str, Enum):
    PENDING = "PENDING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class ImageVectorAnnotationType(str, Enum):
    BBOX = "bbox"
    POLYGON = "polygon"
    POLYLINE = "polyline"
    POINT = "point"


class PointcloudCuboidAnnotationType(str, Enum):
    CUBOID = "cuboid"
    CUBOID_SYNC = "cuboid-sync"


class PointcloudVectorAnnotationType(str, Enum):
    POLYGON = "polygon"
    POLYLINE = "polyline"
    POINT = "point"


class ExportFormat(str, Enum):
    COCO_PANOPTIC = "coco-panoptic"
    COCO_INSTANCE = "coco-instance"
    YOLO = "yolo"
    INSTANCE = "instance"
    INSTANCE_COLOR = "instance-color"
    SEMANTIC = "semantic"
    SEMANTIC_COLOR = "semantic-color"
    POLYGON = "polygon"


class Subscription(str, Enum):
    FREE = "FREE"
    STANDARD = "STANDARD"
    ENTERPRISE = "ENTERPRISE"
    ACADEMIC = "ACADEMIC"
    TRIAL = "TRIAL"


RGB: TypeAlias = "tuple[int, int, int]"
RGBA: TypeAlias = "tuple[int, int, int, int]"
FormatVersion: TypeAlias = "float | str"
ObjectAttributes: TypeAlias = "dict[str, str | bool | int | None]"
ImageAttributes: TypeAlias = "dict[str, str | bool | int | None]"


###########
# Release #
###########
class URL(BaseModel):
    # TODO Remove optional (the backend does not return an URL when adding a release)
    url: str | None = None


class Release(BaseModel):
    uuid: str
    name: str
    description: str
    release_type: Literal["JSON"]
    attributes: URL
    status: ReleaseStatus
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
    comments: list[IssueComment]
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
    attributes: ObjectAttributes | None = None


# Image segmentation
class ImageSegmentationLabelAttributes(BaseModel):
    annotations: list[Annotation]
    segmentation_bitmap: URL
    image_attributes: ImageAttributes | None = None
    format_version: FormatVersion | None = None


# Image vector
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class ImageVectorAnnotation(BaseModel):
    id: int
    category_id: int
    points: list[list[float]]
    type: ImageVectorAnnotationType
    attributes: ObjectAttributes | None = None


class ImageVectorLabelAttributes(BaseModel):
    annotations: list[ImageVectorAnnotation]
    format_version: FormatVersion | None = None
    image_attributes: ImageAttributes | None = None


# Image sequence segmentation
class ImageSequenceSegmentationAnnotation(Annotation):
    track_id: int
    is_keyframe: bool = False


class ImageSequenceSegmentationFrame(ImageSegmentationLabelAttributes):
    annotations: list[ImageSequenceSegmentationAnnotation]
    timestamp: str | int | None
    format_version: FormatVersion | None


class ImageSequenceSegmentationLabelAttributes(BaseModel):
    frames: list[ImageSequenceSegmentationFrame]
    format_version: FormatVersion | None


# Image sequence vector
class ImageSequenceVectorAnnotation(ImageVectorAnnotation):
    track_id: int
    is_keyframe: bool = False


class ImageVectorFrame(ImageVectorLabelAttributes):
    annotations: list[ImageSequenceVectorAnnotation]
    timestamp: str | int | None = None
    format_version: FormatVersion | None = None
    image_attributes: ImageAttributes | None = None


class ImageSequenceVectorLabelAttributes(BaseModel):
    frames: list[ImageVectorFrame]
    format_version: FormatVersion | None = None


# Point cloud segmentation
class PointcloudSegmentationLabelAttributes(BaseModel):
    annotations: list[Annotation]
    point_annotations: list[int]
    format_version: FormatVersion | None = None


class XYZ(BaseModel):
    x: float
    y: float
    z: float


class XYZW(BaseModel):
    qx: float
    qy: float
    qz: float
    qw: float


class FisheyeDistortionCoefficients(BaseModel):
    k1: float
    k2: float
    k3: float
    k4: float


class BrownConradyDistortionCoefficients(BaseModel):
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float


class Distortion(BaseModel):
    model: CameraDistortionModel
    coefficients: FisheyeDistortionCoefficients | BrownConradyDistortionCoefficients


# Point cloud cuboid
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class PointcloudCuboidAnnotation(BaseModel):
    id: int
    category_id: int
    position: XYZ
    dimensions: XYZ
    yaw: float
    rotation: XYZW | None = None
    type: PointcloudCuboidAnnotationType
    attributes: ObjectAttributes | None = None


class PointcloudCuboidLabelAttributes(BaseModel):
    annotations: list[PointcloudCuboidAnnotation]
    format_version: FormatVersion | None = None


# Point cloud vector
class PointcloudVectorAnnotation(BaseModel):
    id: int
    category_id: int
    points: list[list[float]]
    type: PointcloudVectorAnnotationType
    attributes: ObjectAttributes | None = None


class PointcloudVectorLabelAttributes(BaseModel):
    annotations: list[PointcloudVectorAnnotation]
    format_version: FormatVersion | None = None


# Point cloud sequence segmentation
class PointcloudSequenceSegmentationAnnotation(Annotation):
    track_id: int
    is_keyframe: bool = False
    attributes: ObjectAttributes | None = None


class PointcloudSegmentationFrame(PointcloudSegmentationLabelAttributes):
    annotations: list[PointcloudSequenceSegmentationAnnotation]
    point_annotations: list[int]
    timestamp: str | int | None = None
    format_version: FormatVersion | None = None


class PointcloudSequenceSegmentationLabelAttributes(BaseModel):
    frames: list[PointcloudSegmentationFrame]
    format_version: FormatVersion | None = None


# Point cloud sequence cuboid
class PointcloudSequenceCuboidAnnotation(PointcloudCuboidAnnotation):
    track_id: int
    is_keyframe: bool = False


class PointcloudSequenceCuboidFrame(PointcloudCuboidLabelAttributes):
    annotations: list[PointcloudSequenceCuboidAnnotation]
    timestamp: str | int | None = None
    format_version: FormatVersion | None = None


class PointcloudSequenceCuboidLabelAttributes(BaseModel):
    frames: list[PointcloudSequenceCuboidFrame]
    format_version: FormatVersion | None = None


# Point cloud sequence vector
class PointcloudSequenceVectorAnnotation(PointcloudVectorAnnotation):
    track_id: int
    is_keyframe: bool = False


class PointcloudSequenceVectorFrame(PointcloudVectorLabelAttributes):
    annotations: list[PointcloudSequenceVectorAnnotation]
    format_version: FormatVersion | None = None
    timestamp: str | int | None = None


class PointcloudSequenceVectorLabelAttributes(BaseModel):
    frames: list[PointcloudSequenceVectorFrame]
    format_version: FormatVersion | None = None


# Multi-sensor
class MultiSensorPointcloudSequenceCuboidLabelAttributes(BaseModel):
    name: str
    task_type: Literal[TaskType.POINTCLOUD_CUBOID_SEQUENCE]
    attributes: PointcloudSequenceCuboidLabelAttributes


class MultiSensorImageSequenceVectorLabelAttributes(BaseModel):
    name: str
    task_type: Literal[TaskType.IMAGE_VECTOR_SEQUENCE]
    attributes: ImageSequenceVectorLabelAttributes


class MultiSensorLabelAttributes(BaseModel):
    sensors: list[
        MultiSensorPointcloudSequenceCuboidLabelAttributes
        | MultiSensorImageSequenceVectorLabelAttributes
    ]


# Text
class TextAnnotation(BaseModel):
    start: int
    end: int
    category_id: int


class TextLabelAttributes(BaseModel):
    annotations: list[TextAnnotation]
    format_version: FormatVersion | None = None


# https://pydantic-docs.helpmanual.io/usage/types/#unions
LabelAttributes: TypeAlias = "ImageVectorLabelAttributes | ImageSegmentationLabelAttributes | ImageSequenceVectorLabelAttributes | ImageSequenceSegmentationLabelAttributes | PointcloudCuboidLabelAttributes | PointcloudVectorLabelAttributes | PointcloudSegmentationLabelAttributes |PointcloudSequenceCuboidLabelAttributes | PointcloudSequenceVectorLabelAttributes | PointcloudSequenceSegmentationLabelAttributes | MultiSensorLabelAttributes | TextLabelAttributes"


class Label(BaseModel):
    sample_uuid: str
    label_type: TaskType
    label_status: LabelStatus
    labelset: str
    attributes: LabelAttributes | None = None
    created_at: str
    created_by: str
    updated_at: str
    score: float | None = None
    rating: float | None = None
    reviewed_at: str | None = None
    reviewed_by: str | None = None


##########
# Sample #
##########
# Image
class ImageSampleAttributes(BaseModel):
    image: URL


# Image sequence
class ImageFrame(ImageSampleAttributes):
    name: str | None = None


class ImageSequenceSampleAttributes(BaseModel):
    frames: list[ImageFrame]


# Point cloud
class PCD(BaseModel):
    url: str
    signed_url: str | None = None
    type: PCDType


class EgoPose(BaseModel):
    position: XYZ
    heading: XYZW


class CameraIntrinsics(BaseModel):
    intrinsic_matrix: list[list[float]]


class CameraExtrinsics(BaseModel):
    translation: XYZ
    rotation: XYZW


class CalibratedImage(URL):
    row: int | None = None
    col: int | None = None
    intrinsics: CameraIntrinsics | None = None
    extrinsics: CameraExtrinsics | None = None
    distortion: Distortion | None = None
    camera_convention: CameraConvention = CameraConvention.OPEN_GL


class PointcloudSampleAttributes(BaseModel):
    pcd: PCD
    images: list[CalibratedImage] | None = None
    ego_pose: EgoPose | None = None
    default_z: float | None = None
    name: str | None = None
    timestamp: str | int | None = None


# Point cloud sequence
class PointcloudSequenceSampleAttributes(BaseModel):
    frames: list[PointcloudSampleAttributes]


# Multi-sensor
class MultiSensorPointcloudSequenceSampleAttributes(BaseModel):
    name: str
    task_type: Literal[TaskType.POINTCLOUD_CUBOID_SEQUENCE]
    attributes: PointcloudSequenceSampleAttributes


class MultiSensorImageSequenceSampleAttributes(BaseModel):
    name: str
    task_type: Literal[TaskType.IMAGE_VECTOR_SEQUENCE]
    attributes: ImageSequenceSampleAttributes


class MultiSensorSampleAttributes(BaseModel):
    sensors: list[
        MultiSensorPointcloudSequenceSampleAttributes
        | MultiSensorImageSequenceSampleAttributes
    ]


# Text
class TextSampleAttributes(BaseModel):
    text: str


SampleAttributes: TypeAlias = "ImageSampleAttributes | ImageSequenceSampleAttributes | PointcloudSampleAttributes | PointcloudSequenceSampleAttributes | MultiSensorSampleAttributes | TextSampleAttributes "


class Sample(BaseModel):
    uuid: str
    name: str
    attributes: SampleAttributes
    metadata: dict[str, Any]
    created_at: str
    created_by: str
    assigned_labeler: str | None = None
    assigned_reviewer: str | None = None
    comments: list[str] | None = None
    priority: float
    has_embedding: bool | None = None
    label: Label | None = None
    issues: list[Issue] | None = None
    dataset_full_name: str | None = None


########################
# Dataset and labelset #
########################
# https://docs.pydantic.dev/latest/concepts/postponed_annotations/#self-referencing-or-recursive-models
class User(BaseModel):
    username: str
    created_at: str
    is_organization: bool
    email: str | None = None
    webhooks_enabled: bool | None = None
    private_upload_count: int | None = None
    public_upload_count: int | None = None
    subscription: Subscription | None = None
    is_trial_expired: bool | None = None
    organizations: list[User] | None = None
    organization_created_by: str | None = None
    organization_role: Role | None = None
    members: list[User] | None = None
    insights_urls: dict[str, str] | None = None


class Collaborator(BaseModel):
    user: User
    role: Role


class SelectTaskAttribute(BaseModel):
    name: str
    input_type: Literal[InputType.SELECT]
    values: list[str]
    default_value: str | None = None
    is_mandatory: bool | None = None


class TextTaskAttribute(BaseModel):
    name: str
    input_type: Literal[InputType.TEXT] = None
    default_value: str | None = None
    is_mandatory: bool | None = None


class NumberTaskAttribute(BaseModel):
    name: str
    input_type: Literal[InputType.NUMBER]
    default_value: float | None = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    is_mandatory: bool | None = None

    @field_validator("min", "max", "step", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        # min, max and step are empty strings when not filled in
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class CheckboxTaskAttribute(BaseModel):
    name: str
    input_type: Literal[InputType.CHECKBOX]
    default_value: bool | None = None


TaskAttribute: TypeAlias = "SelectTaskAttribute | TextTaskAttribute | NumberTaskAttribute | CheckboxTaskAttribute"


class TaskAttributeCategory(BaseModel):
    name: str
    id: int
    color: RGB | RGBA | None = None
    has_instances: bool | None = None
    attributes: list[TaskAttribute] | None = None
    dimensions: XYZ | None = None
    model_config = ConfigDict(extra="allow")


class TaskAttributes(BaseModel):
    format_version: FormatVersion | None = None
    categories: list[TaskAttributeCategory] | None = None
    image_attributes: list[TaskAttribute] | None = None
    model_config = ConfigDict(extra="allow")


class Owner(BaseModel):
    username: str
    created_at: str
    email: str | None = None


# class Statistics(BaseModel):  # deprecated
#     prelabeled_count: int
#     labeled_count: int
#     reviewed_count: int
#     rejected_count: int
#     skipped_count: int
#     samples_count: int


class Labelset(BaseModel):
    name: str
    description: str
    # uuid: str | None
    # readme: str | None
    # task_type: TaskType | None
    # attributes: str | TaskAttributes | None
    is_groundtruth: bool | None = None
    # statistics: Statistics | None
    created_at: str | None = None
    # stats: dict[str, Any] | None


class Dataset(BaseModel):
    name: str
    full_name: str
    cloned_from: str | None = None
    description: str
    # data_type: Literal["IMAGE"]
    category: Category
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
    labeling_inactivity_timeout_seconds: int | None = None
    samples_count: str | int | None = None
    collaborators_count: int | None = None
    task_attributes: TaskAttributes | None = None
    labelsets: list[Labelset] | None = None
    role: str | None = None
    readme: str | None = None
    metadata: dict[str, Any]
    noncollaborator_can_label: bool | None = None
    noncollaborator_can_review: bool | None = None
    insights_urls: dict[str, str] | None = None
    # tasks: list[dict[str, Any]] | None
    embeddings_enabled: bool | None = None

    @field_validator("category")
    @classmethod
    def check_category(cls, category: str) -> str:
        if category not in Category and "custom-" not in category:
            raise ValidationError(
                f"The category should be one of {Category}, but is {category}."
            )
        return category


####################
# Segments dataset #
####################
class SegmentsDatasetCategory(BaseModel):
    id: int
    name: str
    color: RGB | RGBA | None = None
    attributes: list[Any] | None = None
    model_config = ConfigDict(populate_by_name=True, extra="allow")
