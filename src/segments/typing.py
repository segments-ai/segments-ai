from __future__ import annotations

from enum import Enum as BaseEnum
from enum import EnumMeta as BaseEnumMeta
from typing import Annotated, Any, Dict, List, Optional, Tuple, Union

import pydantic
from pydantic import BaseModel as PydanticBaseModel
from pydantic import ConfigDict, field_validator
from segments.exceptions import ValidationError
from typing_extensions import Literal, TypedDict


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
        return isinstance(item, self) or item in {v.value for v in self.__members__.values()}

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
    REVIEWED: Optional[int] = None
    REVIEWING_IN_PROGRESS: Optional[int] = None
    LABELED: Optional[int] = None
    LABELING_IN_PROGRESS: Optional[int] = None
    REJECTED: Optional[int] = None
    PRELABELED: Optional[int] = None
    SKIPPED: Optional[int] = None
    VERIFIED: Optional[int] = None
    UNLABELED: Optional[int] = None
    # extra
    TOTAL: Optional[int] = None


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
    MULTISENSOR_SEQUENCE = (
        "multisensor-sequence"  # combination of pointcloud-cuboid-sequence and image-vector-sequence
    )
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
    MULTISELECT = "multiselect"
    TEXT = "text"
    NUMBER = "number"
    CHECKBOX = "checkbox"
    VECTOR3 = "vector3"
    QUATERNION = "quaternion"
    POINTS = "points"


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
    LAS = "las"
    SPLAT = "splat"


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


PointcloudCuboidAnnotationTypeLiteral = Literal[
    PointcloudCuboidAnnotationType.CUBOID,
    PointcloudCuboidAnnotationType.CUBOID_SYNC,
]


class PointcloudVectorAnnotationType(str, Enum):
    POLYGON = "polygon"
    POLYLINE = "polyline"
    POINT = "point"


PointcloudVectorAnnotationTypeLiteral = Literal[
    PointcloudVectorAnnotationType.POLYGON,
    PointcloudVectorAnnotationType.POLYLINE,
    PointcloudVectorAnnotationType.POINT,
]


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


RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]
FormatVersion = Union[float, str]
ObjectAttributes = Dict[str, Optional[Union[str, bool, int, float]]]
ImageAttributes = Dict[str, Optional[Union[str, bool, int, float]]]
LinkAttributes = Dict[str, Optional[Union[str, bool, int, float]]]
Timestamp = Union[str, int, float]


###########
# Release #
###########
class URL(BaseModel):
    # TODO Remove optional (the backend does not return an URL when adding a release)
    url: Optional[str] = None
    signed_url: Optional[str] = None


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
    attributes: Optional[ObjectAttributes] = None


class Link(BaseModel):
    from_id: int
    to_id: int
    attributes: Optional[LinkAttributes] = None


# Image segmentation
class ImageSegmentationLabelAttributes(BaseModel):
    annotations: List[Annotation]
    segmentation_bitmap: URL
    image_attributes: Optional[ImageAttributes] = None
    format_version: Optional[FormatVersion] = None


# Image vector
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class ImageVectorAnnotation(BaseModel):
    id: int
    category_id: int
    points: List[List[float]]
    type: ImageVectorAnnotationType
    attributes: Optional[ObjectAttributes] = None


class ImageVectorLabelAttributes(BaseModel):
    annotations: List[ImageVectorAnnotation]
    format_version: Optional[FormatVersion] = None
    image_attributes: Optional[ImageAttributes] = None


# Image sequence segmentation
class ImageSequenceSegmentationAnnotation(Annotation):
    track_id: int
    is_keyframe: bool = False


class ImageSequenceSegmentationFrame(ImageSegmentationLabelAttributes):
    annotations: List[ImageSequenceSegmentationAnnotation]
    timestamp: Optional[Timestamp] = None
    format_version: Optional[FormatVersion] = None


class ImageSequenceSegmentationLabelAttributes(BaseModel):
    frames: List[ImageSequenceSegmentationFrame]
    format_version: Optional[FormatVersion] = None


# Image sequence vector
class ImageSequenceVectorAnnotation(ImageVectorAnnotation):
    track_id: int
    is_keyframe: bool = False


class ImageVectorFrame(ImageVectorLabelAttributes):
    annotations: List[ImageSequenceVectorAnnotation]
    timestamp: Optional[Timestamp] = None
    format_version: Optional[FormatVersion] = None
    image_attributes: Optional[ImageAttributes] = None


class ImageSequenceVectorLabelAttributes(BaseModel):
    frames: List[ImageVectorFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud segmentation
class PointcloudSegmentationLabelAttributes(BaseModel):
    annotations: List[Annotation]
    image_attributes: Optional[ImageAttributes] = None
    point_annotations: List[int]
    format_version: Optional[FormatVersion] = None


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
    k4: Optional[float] = None
    k5: Optional[float] = None
    k6: Optional[float] = None


class DistortionFisheye(BaseModel):
    model: Literal[CameraDistortionModel.FISH_EYE]
    coefficients: FisheyeDistortionCoefficients


class DistortionBrownConrady(BaseModel):
    model: Literal[CameraDistortionModel.BROWN_CONRADY]
    coefficients: BrownConradyDistortionCoefficients


Distortion = Annotated[Union[DistortionFisheye, DistortionBrownConrady], pydantic.Field(discriminator="model")]


# Point cloud cuboid
# https://stackoverflow.com/questions/51575931/class-inheritance-in-python-3-7-dataclasses
class PointcloudCuboidAnnotation(BaseModel):
    id: int
    category_id: int
    position: XYZ
    dimensions: XYZ
    yaw: float
    rotation: Optional[XYZW] = None
    type: PointcloudCuboidAnnotationTypeLiteral
    attributes: Optional[ObjectAttributes] = None


class BasePointcloudCuboidLabelAttributes(BaseModel):
    annotations: List[PointcloudCuboidAnnotation]
    image_attributes: Optional[ImageAttributes] = None
    format_version: Optional[FormatVersion] = None


class PointcloudCuboidLabelAttributes(BasePointcloudCuboidLabelAttributes):
    links: Optional[List[Link]] = None


# Point cloud vector
class PointcloudVectorAnnotation(BaseModel):
    id: int
    category_id: int
    points: List[List[float]]
    type: PointcloudVectorAnnotationTypeLiteral
    attributes: Optional[ObjectAttributes] = None


class BasePointcloudVectorLabelAttributes(BaseModel):
    annotations: List[
        Annotated[Union[PointcloudVectorAnnotation, PointcloudCuboidAnnotation], pydantic.Field(discriminator="type")]
    ]
    image_attributes: Optional[ImageAttributes] = None
    format_version: Optional[FormatVersion] = None


class PointcloudVectorLabelAttributes(BasePointcloudVectorLabelAttributes):
    links: Optional[List[Link]] = None


# Point cloud sequence segmentation
class PointcloudSequenceSegmentationAnnotation(Annotation):
    track_id: int
    is_keyframe: bool = False
    attributes: Optional[ObjectAttributes] = None


class PointcloudSegmentationFrame(PointcloudSegmentationLabelAttributes):
    annotations: List[PointcloudSequenceSegmentationAnnotation]
    timestamp: Optional[Timestamp] = None


class PointcloudSequenceSegmentationLabelAttributes(BaseModel):
    frames: List[PointcloudSegmentationFrame]
    format_version: Optional[FormatVersion] = None


# Point cloud sequence cuboid
class PointcloudSequenceCuboidAnnotation(PointcloudCuboidAnnotation):
    track_id: int
    is_keyframe: bool = False


class PointcloudSequenceCuboidFrame(BasePointcloudCuboidLabelAttributes):
    annotations: List[PointcloudSequenceCuboidAnnotation]
    timestamp: Optional[Timestamp] = None
    format_version: Optional[FormatVersion] = None


class BasePointcloudSequenceCuboidLabelAttributes(BaseModel):
    frames: List[PointcloudSequenceCuboidFrame]
    format_version: Optional[FormatVersion] = None


class PointcloudSequenceCuboidLabelAttributes(BasePointcloudSequenceCuboidLabelAttributes):
    links: Optional[List[Link]] = None


# Point cloud sequence vector
class PointcloudSequenceVectorAnnotation(PointcloudVectorAnnotation):
    track_id: int
    is_keyframe: bool = False


class PointcloudSequenceVectorFrame(PointcloudVectorLabelAttributes):
    annotations: List[
        Annotated[
            Union[PointcloudSequenceVectorAnnotation, PointcloudSequenceCuboidAnnotation],
            pydantic.Field(discriminator="type"),
        ]
    ]
    timestamp: Optional[Timestamp] = None


class BasePointcloudSequenceVectorLabelAttributes(BaseModel):
    frames: List[PointcloudSequenceVectorFrame]
    format_version: Optional[FormatVersion] = None


class PointcloudSequenceVectorLabelAttributes(BasePointcloudSequenceVectorLabelAttributes):
    links: Optional[List[Link]] = None


# Multi-sensor
class MultiSensorPointcloudSequenceCuboidLabelAttributes(BaseModel):
    name: str
    task_type: Literal[TaskType.POINTCLOUD_CUBOID_SEQUENCE]
    # TODO remove list and replace with `Optional[PointcloudSequenceCuboidLabelAttributes] = None`
    attributes: Union[BasePointcloudSequenceCuboidLabelAttributes, List]


class MultiSensorImageSequenceVectorLabelAttributes(BaseModel):
    name: str
    task_type: Literal[TaskType.IMAGE_VECTOR_SEQUENCE]
    # TODO remove list and replace with `Optional[ImageSequenceVectorLabelAttributes] = None`
    attributes: Union[ImageSequenceVectorLabelAttributes, List]


class MultiSensorPointcloudSequenceVectorLabelAttributes(BaseModel):
    name: str
    task_type: Literal[TaskType.POINTCLOUD_VECTOR_SEQUENCE]
    # TODO remove list and replace with `Optional[ImageSequenceVectorLabelAttributes] = None`
    attributes: Union[BasePointcloudSequenceVectorLabelAttributes, List]


class MultiSensorLabelAttributes(BaseModel):
    sensors: List[
        Annotated[
            Union[
                MultiSensorPointcloudSequenceCuboidLabelAttributes,
                MultiSensorPointcloudSequenceVectorLabelAttributes,
                MultiSensorImageSequenceVectorLabelAttributes,
            ],
            pydantic.Field(discriminator="task_type"),
        ]
    ]
    links: Optional[List[Link]] = None


LabelAttributes = Union[
    ImageVectorLabelAttributes,
    ImageSegmentationLabelAttributes,
    ImageSequenceVectorLabelAttributes,
    ImageSequenceSegmentationLabelAttributes,
    PointcloudCuboidLabelAttributes,
    PointcloudVectorLabelAttributes,
    PointcloudSegmentationLabelAttributes,
    PointcloudSequenceCuboidLabelAttributes,
    PointcloudSequenceVectorLabelAttributes,
    PointcloudSequenceSegmentationLabelAttributes,
    MultiSensorLabelAttributes,
]


class Label(BaseModel):
    sample_uuid: str
    label_type: TaskType
    label_status: LabelStatus
    labelset: str
    attributes: Optional[LabelAttributes] = None
    created_at: str
    created_by: str
    updated_at: str
    updated_by: Optional[str] = None
    score: Optional[float] = None
    rating: Optional[float] = None
    reviewed_at: Optional[str] = None
    reviewed_by: Optional[str] = None


class LabelSummary(BaseModel):
    score: Optional[float] = None
    label_status: LabelStatus
    updated_at: Optional[str] = None
    created_by: Optional[str] = None
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
    timestamp: Optional[Timestamp] = None


class ImageSequenceSampleAttributes(BaseModel):
    frames: List[ImageFrame]


# Point cloud
class PCD(BaseModel):
    url: str
    signed_url: Optional[str] = None
    type: PCDType


class EgoPose(BaseModel):
    position: XYZ
    heading: XYZW


class Bounds(BaseModel):
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None
    min_z: Optional[float] = None
    max_z: Optional[float] = None


class CameraIntrinsics(BaseModel):
    intrinsic_matrix: List[List[float]]


class CameraExtrinsics(BaseModel):
    translation: XYZ
    rotation: XYZW


class CalibratedImage(URL):
    row: Optional[int] = None
    col: Optional[int] = None
    intrinsics: Optional[CameraIntrinsics] = None
    extrinsics: Optional[CameraExtrinsics] = None
    distortion: Optional[Distortion] = None
    camera_convention: CameraConvention = CameraConvention.OPEN_GL
    name: Optional[str] = None
    rotation: Optional[float] = None


class PointcloudSampleAttributes(BaseModel):
    pcd: PCD
    images: List[CalibratedImage] = []
    ego_pose: Optional[EgoPose] = None
    default_z: Optional[float] = None
    name: Optional[str] = None
    timestamp: Optional[Timestamp] = None
    bounds: Optional[Bounds] = None


# Point cloud sequence
class PointcloudSequenceSampleAttributes(BaseModel):
    frames: List[PointcloudSampleAttributes]


# Multi-sensor
class MultiSensorPointcloudSequenceSampleAttributes(BaseModel):
    name: str
    task_type: Union[
        Literal[TaskType.POINTCLOUD_CUBOID_SEQUENCE],
        Literal[TaskType.POINTCLOUD_VECTOR_SEQUENCE],
    ]
    attributes: PointcloudSequenceSampleAttributes


class MultiSensorImageSequenceSampleAttributes(BaseModel):
    name: str
    task_type: Literal[TaskType.IMAGE_VECTOR_SEQUENCE]
    attributes: ImageSequenceSampleAttributes


class MultiSensorSampleAttributes(BaseModel):
    sensors: List[
        Union[
            MultiSensorPointcloudSequenceSampleAttributes,
            MultiSensorImageSequenceSampleAttributes,
        ],
    ]


SampleAttributes = Union[
    ImageSampleAttributes,
    ImageSequenceSampleAttributes,
    PointcloudSampleAttributes,
    PointcloudSequenceSampleAttributes,
    MultiSensorSampleAttributes,
]


class Sample(BaseModel):
    uuid: str
    name: str
    attributes: Optional[SampleAttributes] = None
    metadata: Dict[str, Any]
    created_at: str
    created_by: str
    readme: str = ""
    assigned_labeler: Optional[str] = None
    assigned_reviewer: Optional[str] = None
    priority: float
    label: Optional[Union[Label, LabelSummary]] = None
    issues: Optional[List[Issue]] = None
    dataset_full_name: Optional[str] = None


########################
# Dataset and labelset #
########################
# https://docs.pydantic.dev/latest/concepts/postponed_annotations/#self-referencing-or-recursive-models
class User(BaseModel):
    username: str
    created_at: str
    is_organization: bool
    email: Optional[str] = None
    webhooks_enabled: Optional[bool] = None
    private_upload_count: Optional[int] = None
    public_upload_count: Optional[int] = None
    subscription: Optional[Subscription] = None
    is_trial_expired: Optional[bool] = None
    organizations: Optional[List[User]] = None
    organization_created_by: Optional[str] = None
    organization_role: Optional[Role] = None
    members: Optional[List[User]] = None
    insights_urls: Optional[Dict[str, str]] = None


class Collaborator(BaseModel):
    user: User
    role: Role


class BaseAttribute(BaseModel):
    name: str
    is_mandatory: Optional[bool] = None


class BaseSelectTaskAttribute(BaseAttribute):
    input_type: Literal[InputType.SELECT]
    values: List[str]
    default_value: Optional[str] = None


class BaseMultiselectTaskAttribute(BaseAttribute):
    input_type: Literal[InputType.MULTISELECT]
    values: List[str]
    default_value: Optional[str] = None


class BaseTextTaskAttribute(BaseAttribute):
    input_type: Literal[InputType.TEXT]
    default_value: Optional[str] = None


class BaseNumberTaskAttribute(BaseAttribute):
    input_type: Literal[InputType.NUMBER]
    default_value: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None

    @field_validator("min", "max", "step", mode="before")
    @classmethod
    def empty_str_to_none(cls, v):
        # `min`, `max` and `step` are empty strings when not filled in
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


class BaseCheckboxTaskAttribute(BaseAttribute):
    input_type: Literal[InputType.CHECKBOX]
    default_value: Optional[bool] = None


class BaseVector3TaskAttribute(BaseAttribute):
    input_type: Literal[InputType.VECTOR3]


class BaseQuaternionTaskAttribute(BaseAttribute):
    input_type: Literal[InputType.QUATERNION]


class BasePointsTaskAttribute(BaseAttribute):
    input_type: Literal[InputType.POINTS]


class BaseTaskAttribute(BaseModel):
    is_track_level: Optional[bool] = None
    synced_across_sensors: Optional[bool] = None
    sensor_filter: Optional[str] = None


class SelectTaskAttribute(BaseTaskAttribute, BaseSelectTaskAttribute):
    pass


class MultiselectTaskAttribute(BaseTaskAttribute, BaseMultiselectTaskAttribute):
    pass


class TextTaskAttribute(BaseTaskAttribute, BaseTextTaskAttribute):
    pass


class NumberTaskAttribute(BaseTaskAttribute, BaseNumberTaskAttribute):
    pass


class CheckboxTaskAttribute(BaseTaskAttribute, BaseCheckboxTaskAttribute):
    pass


class Vector3TaskAttribute(BaseTaskAttribute, BaseVector3TaskAttribute):
    pass


class QuaternionTaskAttribute(BaseTaskAttribute, BaseQuaternionTaskAttribute):
    pass


class PointsTaskAttribute(BaseTaskAttribute, BasePointsTaskAttribute):
    pass


TaskAttribute = Annotated[
    Union[
        SelectTaskAttribute,
        MultiselectTaskAttribute,
        TextTaskAttribute,
        NumberTaskAttribute,
        CheckboxTaskAttribute,
        Vector3TaskAttribute,
        QuaternionTaskAttribute,
        PointsTaskAttribute,
    ],
    pydantic.Field(discriminator="input_type"),
]
"""TaskAttribute can be one of the following:

- :py:class:`SelectTaskAttribute`
- :py:class:`MultiselectTaskAttribute`
- :py:class:`TextTaskAttribute`
- :py:class:`NumberTaskAttribute`
- :py:class:`CheckboxTaskAttribute`
- :py:class:`Vector3TaskAttribute`
- :py:class:`QuaternionTaskAttribute`
- :py:class:`PointsTaskAttribute`
"""


class BaseTaskLinkAttribute(BaseModel):
    is_track_level: Literal[True] = True


class SelectTaskLinkAttribute(BaseTaskLinkAttribute, BaseSelectTaskAttribute):
    pass


class MultiselectTaskLinkAttribute(BaseTaskLinkAttribute, BaseMultiselectTaskAttribute):
    pass


class TextTaskLinkAttribute(BaseTaskLinkAttribute, BaseTextTaskAttribute):
    pass


class NumberTaskLinkAttribute(BaseTaskLinkAttribute, BaseNumberTaskAttribute):
    pass


class CheckboxTaskLinkAttribute(BaseTaskLinkAttribute, BaseCheckboxTaskAttribute):
    pass


TaskLinkAttribute = Annotated[
    Union[
        SelectTaskLinkAttribute,
        MultiselectTaskLinkAttribute,
        TextTaskLinkAttribute,
        NumberTaskLinkAttribute,
        CheckboxTaskLinkAttribute,
    ],
    pydantic.Field(discriminator="input_type"),
]
"""TaskLinkAttribute can be one of the following:


- :py:class:`SelectTaskLinkAttribute`
- :py:class:`MultiselectTaskLinkAttribute`
- :py:class:`TextTaskLinkAttribute`
- :py:class:`NumberTaskLinkAttribute`
- :py:class:`CheckboxTaskLinkAttribute`
"""


class TaskAttributeCategory(BaseModel):
    name: str
    id: int
    color: Optional[Union[RGB, RGBA]] = None
    has_instances: Optional[bool] = None
    attributes: Optional[List[TaskAttribute]] = None
    dimensions: Optional[XYZ] = None
    model_config = ConfigDict(extra="allow")
    link_attributes: Optional[List[TaskLinkAttribute]] = None


class TaskAttributes(BaseModel):
    format_version: Optional[FormatVersion] = None
    categories: Optional[List[TaskAttributeCategory]] = None
    image_attributes: Optional[List[TaskAttribute]] = None
    model_config = ConfigDict(extra="allow")


class Owner(BaseModel):
    username: str
    created_at: str
    email: Optional[str] = None


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


class Workunit(BaseModel):
    uuid: Optional[str] = None
    created_at: str
    created_by: str
    work_type: str
    next_label_status: Optional[str] = None
    time: int
    inactive_time: int
    sample_uuid: str
    sample_name: str
    label_rating: Optional[int] = None


class Dataset(BaseModel):
    name: str
    full_name: str
    cloned_from: Optional[str] = None
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
    use_timestamps_for_interpolation: bool
    enable_object_linking: bool
    task_type: TaskType
    # task_readme: str
    label_stats: LabelStats
    labeling_inactivity_timeout_seconds: Optional[int] = None
    samples_count: Optional[Union[str, int]] = None
    collaborators_count: Optional[int] = None
    task_attributes: Optional[TaskAttributes] = None
    labelsets: Optional[List[Labelset]] = None
    role: Optional[str] = None
    readme: Optional[str] = None
    metadata: Dict[str, Any]
    noncollaborator_can_label: Optional[bool] = None
    noncollaborator_can_review: Optional[bool] = None
    insights_urls: Optional[Dict[str, str]] = None
    # tasks: Optional[List[Dict[str, Any]]] = None

    @field_validator("category")
    @classmethod
    def check_category(cls, category: str) -> str:
        if category not in Category and "custom-" not in category:
            raise ValidationError(f"The category should be one of {Category}, but is {category}.")
        return category


####################
# Segments dataset #
####################
class SegmentsDatasetCategory(BaseModel):
    id: int
    name: str
    color: Optional[Union[RGB, RGBA]] = None
    attributes: Optional[List[Any]] = None
    model_config = ConfigDict(populate_by_name=True, extra="allow")
