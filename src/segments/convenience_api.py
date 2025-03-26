from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import pydantic
from segments.exceptions import InvalidModelError

from . import typing as segments_typing


if TYPE_CHECKING:  # pragma: no cover
    from .client import SegmentsClient


task_type_sample_attribute_map = {
    segments_typing.TaskType.SEGMENTATION_BITMAP: segments_typing.ImageSampleAttributes,
    segments_typing.TaskType.SEGMENTATION_BITMAP_HIGHRES: segments_typing.ImageSampleAttributes,
    segments_typing.TaskType.IMAGE_SEGMENTATION_SEQUENCE: segments_typing.ImageSequenceSampleAttributes,
    segments_typing.TaskType.BBOXES: segments_typing.ImageSampleAttributes,
    segments_typing.TaskType.VECTOR: segments_typing.ImageSampleAttributes,
    segments_typing.TaskType.IMAGE_VECTOR_SEQUENCE: segments_typing.ImageSequenceSampleAttributes,
    segments_typing.TaskType.KEYPOINTS: segments_typing.ImageSampleAttributes,
    segments_typing.TaskType.POINTCLOUD_CUBOID: segments_typing.PointcloudSampleAttributes,
    segments_typing.TaskType.POINTCLOUD_CUBOID_SEQUENCE: segments_typing.PointcloudSequenceSampleAttributes,
    segments_typing.TaskType.POINTCLOUD_SEGMENTATION: segments_typing.PointcloudSampleAttributes,
    segments_typing.TaskType.POINTCLOUD_SEGMENTATION_SEQUENCE: segments_typing.PointcloudSequenceSampleAttributes,
    segments_typing.TaskType.POINTCLOUD_VECTOR: segments_typing.PointcloudSampleAttributes,
    segments_typing.TaskType.POINTCLOUD_VECTOR_SEQUENCE: segments_typing.PointcloudSequenceSampleAttributes,
    segments_typing.TaskType.MULTISENSOR_SEQUENCE: segments_typing.MultiSensorSampleAttributes,
}

task_type_label_attribute_map = {
    segments_typing.TaskType.SEGMENTATION_BITMAP: segments_typing.ImageSegmentationLabelAttributes,
    segments_typing.TaskType.SEGMENTATION_BITMAP_HIGHRES: segments_typing.ImageSegmentationLabelAttributes,
    segments_typing.TaskType.IMAGE_SEGMENTATION_SEQUENCE: segments_typing.ImageSequenceSegmentationLabelAttributes,
    segments_typing.TaskType.BBOXES: segments_typing.ImageVectorLabelAttributes,
    segments_typing.TaskType.VECTOR: segments_typing.ImageVectorLabelAttributes,
    segments_typing.TaskType.IMAGE_VECTOR_SEQUENCE: segments_typing.ImageSequenceVectorLabelAttributes,
    segments_typing.TaskType.KEYPOINTS: segments_typing.ImageVectorLabelAttributes,
    segments_typing.TaskType.POINTCLOUD_CUBOID: segments_typing.PointcloudCuboidLabelAttributes,
    segments_typing.TaskType.POINTCLOUD_CUBOID_SEQUENCE: segments_typing.PointcloudSequenceCuboidLabelAttributes,
    segments_typing.TaskType.POINTCLOUD_SEGMENTATION: segments_typing.PointcloudSegmentationLabelAttributes,
    segments_typing.TaskType.POINTCLOUD_SEGMENTATION_SEQUENCE: segments_typing.PointcloudSequenceSegmentationLabelAttributes,
    segments_typing.TaskType.POINTCLOUD_VECTOR: segments_typing.PointcloudVectorLabelAttributes,
    segments_typing.TaskType.POINTCLOUD_VECTOR_SEQUENCE: segments_typing.PointcloudSequenceVectorLabelAttributes,
    segments_typing.TaskType.MULTISENSOR_SEQUENCE: segments_typing.MultiSensorLabelAttributes,
}


class HasClient(pydantic.BaseModel):
    _client: Optional["SegmentsClient"] = None

    def _inject_client(self, client: "SegmentsClient"):
        self._client = client


class Dataset(segments_typing.Dataset, HasClient):
    def add_sample(
        self,
        name: str,
        attributes: Union[Dict[str, Any], segments_typing.SampleAttributes],
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0,
        assigned_labeler: Optional[str] = None,
        assigned_reviewer: Optional[str] = None,
        readme: str = "",
        enable_compression: bool = True,
    ) -> Sample:
        attributes = validate_sample_attributes(attributes, self.task_type)

        return self._client.add_sample(
            self.full_name,
            name,
            attributes,
            metadata,
            priority,
            assigned_labeler,
            assigned_reviewer,
            readme,
            enable_compression,
        )

    def delete(self) -> None:
        return self._client.delete_dataset(self.full_name)

    def update(
        self,
        description: Optional[str] = None,
        task_type: Optional[segments_typing.TaskType] = None,
        task_attributes: Optional[Union[Dict[str, Any], segments_typing.TaskAttributes]] = None,
        category: Optional[segments_typing.Category] = None,
        public: Optional[bool] = None,
        readme: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        labeling_inactivity_timeout_seconds: Optional[int] = None,
        enable_skip_labeling: Optional[bool] = None,
        enable_skip_reviewing: Optional[bool] = None,
        enable_ratings: Optional[bool] = None,
        enable_interpolation: Optional[bool] = None,
        enable_same_dimensions_track_constraint: Optional[bool] = None,
        enable_save_button: Optional[bool] = None,
        enable_label_status_verified: Optional[bool] = None,
        enable_3d_cuboid_rotation: Optional[bool] = None,
    ) -> Dataset:
        return self._client.update_dataset(
            self.full_name,
            description,
            task_type,
            task_attributes,
            category,
            public,
            readme,
            metadata,
            labeling_inactivity_timeout_seconds,
            enable_skip_labeling,
            enable_skip_reviewing,
            enable_ratings,
            enable_interpolation,
            enable_same_dimensions_track_constraint,
            enable_save_button,
            enable_label_status_verified,
            enable_3d_cuboid_rotation,
        )

    def get_samples(
        self,
        labelset: Optional[str] = None,
        name: Optional[str] = None,
        label_status: Optional[Union[segments_typing.LabelStatus, List[segments_typing.LabelStatus]]] = None,
        metadata: Optional[Union[str, List[str]]] = None,
        sort: Literal["name", "created", "priority", "updated_at", "gt_label__updated_at"] = "name",
        direction: Literal["asc", "desc"] = "asc",
        per_page: int = 1000,
        page: int = 1,
        include_full_label: bool = False,
    ) -> List[Sample]:
        samples = self._client.get_samples(
            self.full_name,
            labelset,
            name,
            label_status,
            metadata,
            sort,
            direction,
            per_page,
            page,
            include_full_label,
        )
        for sample in samples:
            sample._dataset = self

        return samples

    def clone(
        self,
        new_name: Optional[str] = None,
        new_task_type: Optional[segments_typing.TaskType] = None,
        new_public: Optional[bool] = None,
        organization: Optional[str] = None,
        clone_labels: bool = False,
    ):
        return self._client.clone_dataset(
            self.full_name, new_name, new_task_type, new_public, organization, clone_labels
        )

    def get_collaborator(self, username: str) -> segments_typing.Collaborator:
        return self._client.get_dataset_collaborator(self.full_name, username)

    def add_collaborator(
        self, username: str, role: segments_typing.Role = segments_typing.Role.LABELER
    ) -> segments_typing.Collaborator:
        return self._client.add_dataset_collaborator(self.full_name, username, role)

    def add_release(self, name: str, description: str = "") -> segments_typing.Release:
        return self._client.add_release(self.full_name, name, description)

    def get_release(self, name: str) -> segments_typing.Release:
        return self._client.get_release(self.full_name, name)

    def get_releases(self) -> List[segments_typing.Release]:
        return self._client.get_releases(self.full_name)

    def get_labelsets(self) -> List[segments_typing.Labelset]:
        return self._client.get_labelsets(self.full_name)

    def get_labelset(self, name: str) -> segments_typing.Labelset:
        return self._client.get_labelset(self.full_name, name)

    def add_labelset(self, name: str, description: str = "") -> segments_typing.Labelset:
        return self._client.add_labelset(self.full_name, name, description)

    def get_issues(self) -> List[segments_typing.Issue]:
        return self._client.get_issues(self.full_name)

    def get_workunits(
        self,
        sort: str = "created_at",
        direction: str = "desc",
        start: Optional[str] = None,
        end: Optional[str] = None,
        per_page: int = 1000,
        page: int = 1,
    ) -> List[segments_typing.Workunit]:
        return self._client.get_workunits(self.full_name, sort, direction, start, end, per_page, page)


def inject_sample(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        model = func(self, *args, **kwargs)
        model._sample = self
        return model

    return wrapper


class Sample(segments_typing.Sample, HasClient):
    _dataset: Dataset = None

    def delete(self) -> None:
        return self._client.delete_sample(self.uuid)

    @property
    def dataset(self) -> Dataset:
        # Lazy load the dataset. This should be provided as often as possible but might be missing
        # when using the low-level client approach.
        if self._dataset is None:
            if self.dataset_full_name is None:
                s2 = self._client.get_sample(self.uuid)
                self.dataset_full_name = s2.dataset_full_name

            self._dataset = self._client.get_dataset(self.dataset_full_name)

        return self._dataset

    def update(
        self,
        name: Optional[str] = None,
        attributes: Optional[Union[Dict[str, Any], segments_typing.SampleAttributes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
        assigned_labeler: Optional[str] = None,
        assigned_reviewer: Optional[str] = None,
        readme: Optional[str] = None,
        enable_compression: bool = True,
    ) -> Sample:
        attributes_model = validate_sample_attributes(attributes, self.dataset.task_type)
        return self._client.update_sample(
            self.uuid,
            name,
            attributes_model,
            metadata,
            priority,
            assigned_labeler,
            assigned_reviewer,
            readme,
            enable_compression,
        )

    @inject_sample
    def get_label(self, labelset: Optional[str] = "ground-truth") -> Label:
        return self._client.get_label(self.uuid, labelset)

    @inject_sample
    def add_label(
        self,
        labelset: str,
        attributes: Union[Dict[str, Any], segments_typing.LabelAttributes],
        label_status: segments_typing.LabelStatus = segments_typing.LabelStatus.PRELABELED,
        score: Optional[float] = None,
        enable_compression: bool = True,
    ) -> Label:
        attributes_model = validate_label_attributes(attributes, self.dataset.task_type)
        return self._client.add_label(self.uuid, labelset, attributes_model, label_status, score, enable_compression)

    @inject_sample
    def update_label(
        self,
        labelset: str,
        attributes: Union[Dict[str, Any], segments_typing.LabelAttributes],
        label_status: segments_typing.LabelStatus = segments_typing.LabelStatus.PRELABELED,
        score: Optional[float] = None,
        enable_compression: bool = True,
    ) -> Label:
        attributes_model = validate_label_attributes(attributes, self.dataset.task_type)
        return self._client.update_label(
            self.uuid, labelset, attributes_model, label_status, score, enable_compression
        )

    def delete_label(self, labelset: str) -> None:
        return self._client.delete_label(self.uuid, labelset)

    def add_issue(
        self, description: str, status: segments_typing.IssueStatus = segments_typing.IssueStatus.OPEN
    ) -> segments_typing.Issue:
        return self._client.add_issue(self.uuid, description, status)

    def get_issues(self) -> List[segments_typing.Issue]:
        return self._client.get_issues(self.uuid)


class Label(segments_typing.Label, HasClient):
    _sample: Sample = None

    @property
    def sample(self) -> Sample:
        # Lazy load the sample. This should be provided as often as possible but might be missing
        # when using the low-level client approach.
        if self._sample is None:
            self._sample = self._client.get_sample(self.sample_uuid)

        return self._sample

    def update(
        self,
        attributes: Union[Dict[str, Any], segments_typing.LabelAttributes],
        label_status: segments_typing.LabelStatus = segments_typing.LabelStatus.PRELABELED,
        score: Optional[float] = None,
        enable_compression: bool = True,
    ) -> Label:
        attributes = validate_label_attributes(attributes, self.label_type)
        return self._client.update_label(
            self.sample_uuid, self.labelset, attributes, label_status, score, enable_compression
        )

    def delete(self) -> None:
        return self._client.delete_label(self.sample_uuid, self.labelset)


def validate_sample_attributes(attributes, task_type):
    attribute_model = task_type_sample_attribute_map[task_type]
    attributes = _validate_attributes(attributes, task_type, attribute_model)
    return attributes


def validate_label_attributes(attributes, task_type):
    attribute_model = task_type_label_attribute_map[task_type]
    attributes = _validate_attributes(attributes, task_type, attribute_model)
    return attributes


def _validate_attributes(attributes, task_type, attribute_model):
    if isinstance(attributes, dict):
        validator = pydantic.TypeAdapter(attribute_model)
        attributes = validator.validate_python(attributes)
    else:
        if not isinstance(attributes, attribute_model):
            # TODO: Convert to a segments exception
            raise InvalidModelError(f"attributes must be a dict or {attribute_model} for task type {task_type}")
    return attributes
