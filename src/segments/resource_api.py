from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Union

import pydantic
from segments.exceptions import InvalidModelError, MissingContextError

from . import typing as segments_typing
from .sentinel import _NOT_ASSIGNED


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
    """See :class:`~segments.typing.Dataset` an overview of the properties this model has."""

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
        """Adds a sample to the dataset, see :meth:`segments.client.SegmentsClient.add_sample` for more details.

        Args:
            name: The name of the sample.
            attributes: The sample attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/sample-types>`__.
            metadata: Any sample metadata. Example: ``{'weather': 'sunny', 'camera_id': 3}``.
            priority: Priority in the labeling queue. Samples with higher values will be labeled first. Defaults to ``0``.
            assigned_labeler: The username of the user who should label this sample. Leave empty to not assign a specific labeler. Defaults to :obj:`None`.
            assigned_reviewer: The username of the user who should review this sample. Leave empty to not assign a specific reviewer. Defaults to :obj:`None`.
            readme: The sample readme. Defaults to :obj:`None`.
            enable_compression: Whether to enable gzip compression for the request. Defaults to :obj:`True`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the sample attributes fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of the sample fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.AlreadyExistsError`: If the sample already exists.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        attributes = validate_sample_attributes(attributes, self.task_type)

        sample = self._client.add_sample(
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
        sample._dataset = self
        return sample

    def delete(self) -> None:
        """Deletes this dataset. See :meth:`segments.client.SegmentsClient.delete_dataset` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
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
        enable_confirm_on_commit: Optional[bool] = None,
    ) -> Dataset:
        """Updates the dataset, see :meth:`segments.client.SegmentsClient.update_dataset` for more details.

        Args:
            description: The dataset description. Defaults to :obj:`None`.
            task_type: The dataset's task type. Defaults to :obj:`None`.
            task_attributes: The dataset's task attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format>`__. Defaults to :obj:`None`.
            category: The dataset category. Defaults to :obj:`None`.
            public: The dataset visibility. Defaults to :obj:`None`.
            readme: The dataset readme. Defaults to :obj:`None`.
            metadata: Any dataset metadata. Example: ``{'day': 'sunday', 'robot_id': 3}``.
            labeling_inactivity_timeout_seconds: The number of seconds after which a user is considered inactive during labeling. Only impacts label timing metrics. Defaults to :obj:`None`.
            enable_skip_labeling: Enable the skip button in the labeling workflow. Defaults to :obj:`None`.
            enable_skip_reviewing: Enable the skip button in the reviewing workflow. Defaults to :obj:`None`.
            enable_ratings: Enable star-ratings for labeled images. Defaults to :obj:`None`.
            enable_interpolation: Enable label interpolation in sequence datasets. Ignored for non-sequence datasets. Defaults to :obj:`None`.
            enable_same_dimensions_track_constraint: Enable constraint to keep same cuboid dimensions for the entire object track in point cloud cuboid datasets. Ignored for non-cuboid datasets. Defaults to :obj:`None`.
            enable_save_button: Enable a save button in the labeling and reviewing workflow, to save unfinished work. Defaults to :obj:`False`.
            enable_label_status_verified: Enable an additional label status "Verified". Defaults to :obj:`False`.
            enable_3d_cuboid_rotation: Enable 3D cuboid rotation (i.e., yaw, pitch and roll). Defaults to :obj:`False`.
            enable_confirm_on_commit: Enable a confirmation dialog when saving a sample in this dataset. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the dataset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
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
        """Gets all samples within a dataset, see :meth:`segments.client.SegmentsClient.get_samples` for more details.

        Args:
            labelset: If defined, this additionally returns for each sample a label summary or full label (depending on `include_full_label`) for the given labelset. Defaults to :obj:`None`.
            name: Name to filter by. Defaults to :obj:`None` (no filtering).
            label_status: Sequence of label statuses to filter by. Defaults to :obj:`None` (no filtering).
            metadata: Sequence of 'key:value' metadata attributes to filter by. Defaults to :obj:`None` (no filtering).
            sort: What to sort results by. One of ``name``, ``created``, ``priority``, ``updated_at``, ``gt_label__updated_at``. Defaults to ``name``.
            direction: Sorting direction. One of ``asc`` (ascending) or ``desc`` (descending). Defaults to ``asc``.
            per_page: Pagination parameter indicating the maximum number of samples to return. Defaults to ``1000``.
            page: Pagination parameter indicating the page to return. Defaults to ``1``.
            include_full_label: Whether to include the full label in the response, or only a summary. Ignored if `labelset` is `None`. Defaults to :obj:`False`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
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
    ) -> Dataset:
        return self._client.clone_dataset(
            self.full_name, new_name, new_task_type, new_public, organization, clone_labels
        )

    def get_collaborator(self, username: str) -> Collaborator:
        """Fetches a specific collaborator from the dataset, see :meth:`segments.client.SegmentsClient.get_dataset_collaborator` for more details.

        Args:
            username: The username of the collaborator to be added.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the collaborator fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or dataset collaborator is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset collaborator does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        collaborator = self._client.get_dataset_collaborator(self.full_name, username)
        collaborator._dataset = self
        return collaborator

    def add_collaborator(
        self, username: str, role: segments_typing.Role = segments_typing.Role.LABELER
    ) -> Collaborator:
        """Adds a collaborator to the dataset, see :meth:`segments.client.SegmentsClient.add_dataset_collaborator` for more details.

        Args:
            username: The username of the collaborator to be added.
            role: The role of the collaborator to be added. One of ``labeler``, ``reviewer``, ``manager``, ``admin``. Defaults to ``labeler``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the collaborator fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        collaborator = self._client.add_dataset_collaborator(self.full_name, username, role)
        collaborator._dataset = self
        return collaborator

    def add_release(self, name: str, description: str = "") -> segments_typing.Release:
        """Adds a release to the dataset, see :meth:`segments.client.SegmentsClient.add_release` for more details.

        Args:
            name: The name of the release.
            description: The release description. Defaults to ``''``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the release fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.AlreadyExistsError`: If the release already exists.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        release = self._client.add_release(self.full_name, name, description)
        release._dataset = self
        return release

    def get_release(self, name: str) -> Release:
        """Gets a specific release from the dataset, see :meth:`segments.client.SegmentsClient.get_release` for more details.

        Args:
            name: The name of the release.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the release fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or release is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        release = self._client.get_release(self.full_name, name)
        release._dataset = self
        return release

    def get_releases(self) -> List[Release]:
        """Lists all releases in the dataset, see :meth:`segments.client.SegmentsClient.get_releases` for more details.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the releases fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        releases = self._client.get_releases(self.full_name)
        for release in releases:
            release._dataset = self

        return releases

    def get_labelsets(self) -> List[segments_typing.Labelset]:
        """Gets all labelsets in this dataset, see :meth:`segments.client.SegmentsClient.get_labelsets` for more details.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the labelsets fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        labelsets = self._client.get_labelsets(self.full_name)
        for labelset in labelsets:
            labelset._dataset = self

        return labelsets

    def get_labelset(self, name: str) -> segments_typing.Labelset:
        """Gets a labelset from this dataset, see :meth:`segments.client.SegmentsClient.get_labelset` for more details.

        Args:
            name: The name of the labelset.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the labelset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        labelset = self._client.get_labelset(self.full_name, name)
        labelset._dataset = self
        return labelset

    def add_labelset(self, name: str, description: str = "") -> segments_typing.Labelset:
        """Adds a labelset to the dataset, see :meth:`segments.client.SegmentsClient.add_labelset` for more details.

        Args:
            name: The name of the labelset.
            description: The labelset description. Defaults to ``''``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the labelset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.AlreadyExistsError`: If the labelset already exists.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        labelset = self._client.add_labelset(self.full_name, name, description)
        labelset._dataset = self
        return labelset

    def get_issues(self) -> List[Issue]:
        """Gets all issues in this dataset, see :meth:`segments.client.SegmentsClient.get_issues` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
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
        """Lists all workunits associated to this dataset, see :meth:`segments.client.SegmentsClient.get_workunits` for more details.

        Args:
            sort: What to sort results by. One of ``created_at``. Defaults to ``created_at``.
            direction: Sorting direction. One of ``asc`` (ascending) or ``desc`` (descending). Defaults to ``desc``.
            start: The start datetime for filtering workunits. Must be in the format 'YYYY-MM-DDTHH:MM:SS'. Defaults to :obj:`None`.
            end: The end datetime for filtering workunits. Must be in the format 'YYYY-MM-DDTHH:MM:SS'. Defaults to :obj:`None`.
            per_page: Pagination parameter indicating the maximum number of results to return. Defaults to ``1000``.
            page: Pagination parameter indicating the page to return. Defaults to ``1``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        return self._client.get_workunits(self.full_name, sort, direction, start, end, per_page, page)


class Sample(segments_typing.Sample, HasClient):
    """See :class:`~segments.typing.Sample` an overview of the properties this model has."""

    _dataset: Optional[Dataset] = None

    @property
    def dataset(self) -> Dataset:
        """This property contains a lazy loaded dataset object that this sample belongs to.

        When using the high level API, this property is always available. When fetching the sample with a :class:`~SegmentsClient` directly, using this property will trigger an additional API request to fetch the dataset.
        """
        # Lazy load the dataset. This should be provided as often as possible but might be missing
        # when using the low-level client approach.
        if self._dataset is None:
            if self.dataset_full_name is None:
                s2 = self._client.get_sample(self.uuid)
                self.dataset_full_name = s2.dataset_full_name

            self._dataset = self._client.get_dataset(self.dataset_full_name)

        return self._dataset

    def delete(self) -> None:
        """Deletes this sample. See :meth:`segments.client.SegmentsClient.delete_sample` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        return self._client.delete_sample(self.uuid)

    def update(
        self,
        name: Optional[str] = None,
        attributes: Optional[Union[Dict[str, Any], segments_typing.SampleAttributes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
        assigned_labeler: Optional[str] = _NOT_ASSIGNED,
        assigned_reviewer: Optional[str] = _NOT_ASSIGNED,
        readme: Optional[str] = None,
        enable_compression: bool = True,
    ) -> Sample:
        """Updates this sample. See :meth:`segments.client.SegmentsClient.update_sample` for more details.

        Args:
            name: The name of the sample.
            attributes: The sample attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/sample-types>`__.
            metadata: Any sample metadata. Example: ``{'weather': 'sunny', 'camera_id': 3}``.
            priority: Priority in the labeling queue. Samples with higher values will be labeled first.
            assigned_labeler: The username of the user who should label this sample. Leave empty to not assign a specific labeler.
            assigned_reviewer: The username of the user who should review this sample. Leave empty to not assign a specific reviewer.
            readme: The sample readme.
            enable_compression: Whether to enable gzip compression for the request. Defaults to :obj:`True`.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.ValidationError`: If validation of the sample fails.
            :exc:`~segments.exceptions.NotFoundError`: If the sample is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        if attributes is not None:
            attributes = validate_sample_attributes(attributes, self.dataset.task_type)

        return self._client.update_sample(
            self.uuid,
            name,
            attributes,
            metadata,
            priority,
            assigned_labeler,
            assigned_reviewer,
            readme,
            enable_compression,
        )

    def get_label(self, labelset: Optional[str] = "ground-truth") -> Label:
        """Gets the label of this sample. See :meth:`segments.client.SegmentsClient.get_label` for more details.

        Args:
            labelset: The labelset this label belongs to. Defaults to ``ground-truth``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found or if the sample is unlabeled.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        return self._client.get_label(self.uuid, labelset)

    def add_label(
        self,
        labelset: str,
        attributes: Union[Dict[str, Any], segments_typing.LabelAttributes],
        label_status: segments_typing.LabelStatus = segments_typing.LabelStatus.PRELABELED,
        score: Optional[float] = None,
        enable_compression: bool = True,
    ) -> Label:
        """Adds a label to this sample. See :meth:`segments.client.SegmentsClient.add_label` for more details.

        Args:
            labelset: The labelset this label belongs to.
            attributes: The label attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/label-types>`__.
            label_status: The label status. Defaults to ``PRELABELED``.
            score: The label score. Defaults to :obj:`None`.
            enable_compression: Whether to enable gzip compression for the request. Defaults to :obj:`True`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the attributes fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        attributes_model = validate_label_attributes(attributes, self.dataset.task_type)
        return self._client.add_label(self.uuid, labelset, attributes_model, label_status, score, enable_compression)

    def update_label(
        self,
        labelset: str,
        attributes: Optional[Union[Dict[str, Any], segments_typing.LabelAttributes]] = None,
        label_status: segments_typing.LabelStatus = segments_typing.LabelStatus.PRELABELED,
        score: Optional[float] = None,
        enable_compression: bool = True,
    ) -> Label:
        """Updates the label of this sample. See :meth:`segments.client.SegmentsClient.update_label` for more details.

        Args:
            labelset: The labelset this label belongs to.
            attributes: The label attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/label-types>`__. Defaults to :obj:`None`.
            label_status: The label status. Defaults to :obj:`None`.
            score: The label score. Defaults to :obj:`None`.
            enable_compression: Whether to enable gzip compression for the request. Defaults to :obj:`True`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        if attributes is not None:
            attributes = validate_label_attributes(attributes, self.dataset.task_type)

        return self._client.update_label(
            self.uuid, labelset, attributes, label_status, score, enable_compression
        )

    def delete_label(self, labelset: str) -> None:
        """Deletes the label of this sample. See :meth:`segments.client.SegmentsClient.delete_label` for more details.

        Args:
            labelset: The labelset this label belongs to.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        return self._client.delete_label(self.uuid, labelset)

    def add_issue(
        self, description: str, status: segments_typing.IssueStatus = segments_typing.IssueStatus.OPEN
    ) -> Issue:
        """Adds an issue to this sample. See :meth:`segments.client.SegmentsClient.add_issue` for more details.

        Args:
            description: The issue description.
            status: The issue status. One of ``OPEN`` or ``CLOSED``. Defaults to ``OPEN``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the issue fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        return self._client.add_issue(self.uuid, description, status)

    def get_issues(self) -> List[Issue]:
        """Gets all issues associated to this sample. See :meth:`segments.client.SegmentsClient.get_issues` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        return self._client.get_issues(self.uuid)


class Label(segments_typing.Label, HasClient):
    """See :class:`~segments.typing.Label` an overview of the properties this model has."""

    def update(
        self,
        attributes: Optional[Union[Dict[str, Any], segments_typing.LabelAttributes]] = None,
        label_status: segments_typing.LabelStatus = segments_typing.LabelStatus.PRELABELED,
        score: Optional[float] = None,
        enable_compression: bool = True,
    ) -> Label:
        """Updates this sample. See :meth:`segments.client.SegmentsClient.update_label` for more details.

        Args:
            labelset: The labelset this label belongs to.
            attributes: The label attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/label-types>`__. Defaults to :obj:`None`.
            label_status: The label status. Defaults to :obj:`None`.
            score: The label score. Defaults to :obj:`None`.
            enable_compression: Whether to enable gzip compression for the request. Defaults to :obj:`True`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        if attributes is not None:
            attributes = validate_label_attributes(attributes, self.label_type)

        return self._client.update_label(
            self.sample_uuid, self.labelset, attributes, label_status, score, enable_compression
        )

    def delete(self) -> None:
        """Deletes this label. See :meth:`segments.client.SegmentsClient.delete_label` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        return self._client.delete_label(self.sample_uuid, self.labelset)


class Issue(segments_typing.Issue, HasClient):
    """See :class:`~segments.typing.Issue` an overview of the properties this model has."""

    def update(
        self,
        description: Optional[str] = None,
        status: Optional[segments_typing.IssueStatus] = None,
    ) -> Issue:
        """Updates this issue. See :meth:`segments.client.SegmentsClient.add_issue` for more details.

        Args:
            description: The issue description. Defaults to :obj:`None`.
            status: The issue status. One of ``OPEN`` or ``CLOSED``. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the issue fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the issue is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        return self._client.update_issue(self.uuid, description, status)

    def delete(
        self,
    ) -> None:
        """Deletes this issue. See :meth:`segments.client.SegmentsClient.delete_issue` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the issue is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        self._client.delete_issue(self.uuid)


class Collaborator(segments_typing.Collaborator, HasClient):
    """See :class:`~segments.typing.Collaborator` an overview of the properties this model has."""

    _dataset: Optional[Dataset] = None

    @property
    def dataset(self):
        if self._dataset is None:
            raise MissingContextError(
                "It is not possible to deduce which dataset this Collaborator object belongs to. This error occurs when you are not using the resource API. Either fetch the Collaborator using `Dataset.get_collaborator` or use the SegmentsClient directly to operate on the Collaborator resource."
            )

        return self._dataset

    def delete(self) -> None:
        """Delete a dataset collaborator. See :meth:`segments.client.SegmentsClient.delete_dataset_collaborator` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or dataset collaborator is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
            :exc:`~segments.exceptions.MissingContextError`: If the dataset property is not available, usually because the `Collaborator` object was not fetched using the resource api.
        """
        """"""
        self._client.delete_dataset_collaborator(self.dataset.full_name, self.user.username)

    def update(self, role: segments_typing.Role) -> Collaborator:
        """Update this dataset collaborator. See :meth:`segments.client.SegmentsClient.update_dataset_collaborator` for more details.

        Args:
            role: The role of the collaborator to be added. Defaults to ``labeler``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the collaborator fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or dataset collaborator is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
            :exc:`~segments.exceptions.MissingContextError`: If the dataset property is not available, usually because the `Collaborator` object was not fetched using the resource api.
        """
        return self._client.update_dataset_collaborator(self.dataset.full_name, self.user.username, role)


class Release(segments_typing.Release, HasClient):
    _dataset: Optional[Dataset] = None

    @property
    def dataset(self):
        if self._dataset is None:
            raise MissingContextError(
                "It is not possible to deduce which dataset this Release object belongs to. This error occurs when you are not using the resource API. Either fetch the Release using `Dataset.get_release` or use the SegmentsClient directly to operate on the Release resource."
            )

        return self._dataset

    def delete(self) -> None:
        """Deletes this release. See :meth:`segments.client.SegmentsClient.delete_release` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or release is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
            :exc:`~segments.exceptions.MissingContextError`: If the dataset property is not available, usually because the `Release` object was not fetched using the resource api.
        """
        return self._client.delete_release(self.dataset.full_name, self.name)


class Labelset(segments_typing.Labelset, HasClient):
    _dataset: Optional[Dataset] = None

    @property
    def dataset(self):
        if self._dataset is None:
            raise MissingContextError(
                "It is not possible to deduce which dataset this Labelset object belongs to. This error occurs when you are not using the resource API. Either fetch the Labelset using `Dataset.get_labelset` or use the SegmentsClient directly to operate on the Labelset resource."
            )

        return self._dataset

    def delete(self) -> None:
        """Deletes this labelset. See :meth:`segments.client.SegmentsClient.delete_labelset` for more details.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
            :exc:`~segments.exceptions.MissingContextError`: If the dataset property is not available, usually because the `Labelset` object was not fetched using the resource api.
        """
        return self._client.delete_labelset(self.dataset.full_name, self.name)


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
            raise InvalidModelError(f"attributes must be a dict or {attribute_model} for task type {task_type}")
    return attributes
