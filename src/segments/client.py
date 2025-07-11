# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from __future__ import annotations

# https://gist.github.com/benkehoe/066a73903e84576a8d6d911cfedc2df6
import functools
import gzip
import importlib.metadata as importlib_metadata
import inspect
import json
import logging
import os
import urllib.parse
from types import TracebackType
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    List,
    Optional,
    TextIO,
    Type,
    TypeVar,
    Union,
    cast,
)

import pydantic
import requests
from pydantic import TypeAdapter
from segments.exceptions import (
    AlreadyExistsError,
    APILimitError,
    AuthenticationError,
    AuthorizationError,
    CollaboratorError,
    NetworkError,
    NotFoundError,
    SubscriptionError,
    TimeoutError,
    ValidationError,
)
from segments.typing import (
    AWSFields,
    Category,
    File,
    IssueStatus,
    LabelAttributes,
    LabelStatus,
    PresignedPostFields,
    Role,
    SampleAttributes,
    TaskAttributes,
    TaskType,
    User,
    Workunit,
)
from typing_extensions import Literal, get_args

from .resource_api import Collaborator, Dataset, HasClient, Issue, Label, Labelset, Release, Sample
from .sentinel import _NOT_ASSIGNED
from .version import __version__

################################
# Constants and type variables #
################################
logger = logging.getLogger(__name__)
T = TypeVar("T")
VERSION = __version__


####################
# Helper functions #
####################
# Error handling: https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
def handle_exceptions(f: Callable[..., requests.Response]) -> Callable[..., requests.Response]:
    """Catch exceptions and throw Segments exceptions.

    Returns:
        A wrapper function (of this exception handler decorator).

    Raises:
        :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
        :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
        :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
    """

    @functools.wraps(f)
    def throw_segments_exception(
        self: SegmentsClient,
        *args: Any,
        **kwargs: Any,
    ) -> requests.Response:
        try:
            r = f(self, *args, **kwargs)
            r.raise_for_status()
            if r.content:
                r_json = r.json()
                # Check if the API limit is exceeded
                if isinstance(r_json, dict):
                    message = r_json.get("detail", "")
                    if "throttled" in message:
                        raise APILimitError(message)
            return r
        except requests.exceptions.Timeout as e:
            # Maybe set up for a retry, or continue in a retry loop
            raise TimeoutError(message=str(e), cause=e)
        except requests.exceptions.HTTPError as e:
            # Make string comparison case insensitive.
            text = e.response.text.lower()
            if "not found" in text or "does not exist" in text:
                text += """\n
Possible fixes:
- Check for typos in the dataset name, sample name, labelset name, etc.
- Are you using the argument `dataset_name` -> Did you add the organization to the dataset name? E.g., `jane/flowers` instead of `flowers`.
"""
                raise NotFoundError(message=text, cause=e)
            if "already exists" in text or "already have" in text:
                raise AlreadyExistsError(message=text, cause=e)
            if "cannot be added as collaborator" in text or "is already a collaborator" in text:
                raise CollaboratorError(message=text, cause=e)
            if "authentication credentials were not provided" in text:
                raise AuthenticationError(message=text, cause=e)
            if (
                "cannot leave the organization" in text
                or "need to be an administrator" in text
                or "do not have permission" in text
            ):
                raise AuthorizationError(message=text, cause=e)
            if "free trial ended" in text or "exceeded user limit" in text:
                raise SubscriptionError(message=text, cause=e)
            if "time-out" in text:
                raise TimeoutError(message=text, cause=e)
            if "invalid page" in text:
                raise NotFoundError(message=text, cause=e)
            if "502 Server Error: Bad Gateway" in text:
                raise NetworkError(
                    message="502 Server Error. Decrease the `per_page` argument if you called `get_samples`.",
                    cause=e,
                )
            raise NetworkError(message=text, cause=e)
        except requests.exceptions.TooManyRedirects as e:
            # Tell the user their URL was bad and try a different one
            raise NetworkError(message="Bad url, please try a different one.", cause=e)
        except requests.exceptions.RequestException as e:
            logger.error(f"Unknown error: {e}")
            raise NetworkError(message=str(e), cause=e)

    return throw_segments_exception


def convert_model(
    f: Callable[..., requests.Response],
) -> Callable[..., Union[requests.Response, T, dict, list]]:
    """Converts the response of the function to a pydantic model. Which model is determined by the `model` argument.

    Returns:

    Raises:
        :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
    """

    @functools.wraps(f)
    def convert_model_wrapper(
        self: SegmentsClient,
        *args: Any,
        model: Optional[Type[T]] = None,
        **kwargs: Any,
    ):
        resp = f(self, *args, **kwargs)
        if model is None:
            return resp

        r_json = resp.json()
        try:
            m = TypeAdapter(model).validate_python(r_json)

            # Add the client to the model if it has teh HasClient mixin
            # This is required for the resource API to work
            if isinstance(m, list) and issubclass(get_args(model)[0], HasClient):
                for item in m:
                    item._inject_client(self)
            elif inspect.isclass(model) and issubclass(model, HasClient):
                m._inject_client(self)

        except pydantic.ValidationError as e:
            if not self._strict_checking:
                # We're not applying strict type checking, just return the decoded json
                logging.warning(f"Validation failed for model {model}, returning the raw json.")
                return r_json
            else:
                raise ValidationError(message=str(e), cause=e)

        return m

    return convert_model_wrapper


##########
# Client #
##########
class SegmentsClient:
    """A client with a connection to the Segments.ai platform.

    Note:
        Please refer to the `Python SDK quickstart <https://docs.segments.ai/tutorials/python-sdk-quickstart>`__ for a full example of working with the Python SDK.

    First install the SDK.

    .. code-block:: bash

        pip install --upgrade segments-ai

    Import the ``segments`` package in your python file and set up a client with an API key. An API key can be created on your `user account page <https://segments.ai/account>`__.

    .. code-block:: python

        from segments import SegmentsClient
        api_key = 'YOUR_API_KEY'
        client = SegmentsClient(api_key)

    Or store your Segments API key in your environment (``SEGMENTS_API_KEY = 'YOUR_API_KEY'``):

    .. code-block:: python

        from segments import SegmentsClient
        client = SegmentsClient()

    Args:
        api_key: Your Segments.ai API key. If no API key given, reads ``SEGMENTS_API_KEY`` from the environment. Defaults to :obj:`None`.
        api_url: URL of the Segments.ai API. Defaults to ``https://api.segments.ai/``.

    Raises:
        :exc:`~segments.exceptions.AuthenticationError`: If an invalid API key is used or (when not passing the API key directly) if ``SEGMENTS_API_KEY`` is not found in your environment.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://api.segments.ai/",
    ):
        if api_key is None:
            api_key = os.getenv("SEGMENTS_API_KEY")
            if api_key is None:
                raise AuthenticationError(
                    message="Please provide the `api_key` or set `SEGMENTS_API_KEY` in your environment."
                )
            else:
                print("Found a Segments API key in your environment.")

        self.api_key = api_key
        self.api_url = api_url

        # https://realpython.com/python-requests/#performance
        # https://stackoverflow.com/questions/21371809/cleanly-setting-max-retries-on-python-requests-get-or-post-method
        # https://stackoverflow.com/questions/23013220/max-retries-exceeded-with-url-in-requests
        self.api_session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        self.api_session.mount("http://", adapter)
        self.api_session.mount("https://", adapter)

        self.s3_session = requests.Session()
        self.s3_session.mount("http://", adapter)
        self.s3_session.mount("https://", adapter)

        self._strict_checking = True

        try:
            r = self._get(f"/api_status/?lib_version={VERSION}")
            if r.status_code == 200:
                logger.info("Initialized successfully.")
        except NetworkError as e:
            if cast(requests.exceptions.RequestException, e.cause).response.status_code == 426:
                logger.warning(
                    "There's a new version available. Please upgrade by running `pip install --upgrade segments-ai`"
                )
            else:
                raise AuthenticationError(message="Something went wrong. Did you use the right API key?")

    # https://stackoverflow.com/questions/48160728/resourcewarning-unclosed-socket-in-python-3-unit-test
    def close(self) -> None:
        """Close :class:`SegmentsClient` connections.

        You can manually close the Segments client's connections:

        .. code-block:: python

            client = SegmentsClient()
            client.get_datasets()
            client.close()

        Or use the client as a context manager:

        .. code-block:: python

            with SegmentsClient() as client:
                client.get_datasets()
        """
        self.api_session.close()
        self.s3_session.close()
        # logger.info("Closed successfully.")

    # Use SegmentsClient as a context manager (e.g., with SegmentsClient() as client: client.add_dataset()).
    def __enter__(self) -> SegmentsClient:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.close()

    ########
    # User #
    ########
    def get_user(self, user: Optional[str] = None) -> User:
        """Get a user.

        .. code-block:: python

            user = client.get_user()
            print(user)

        Args:
            user: The username for which to get the user details. Leave empty to get the authenticated user. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the user fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the user is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if user is not None:
            r = self._get(f"/users/{user}", model=User)
        else:
            r = self._get("/user", model=User)

        return cast(User, r)

    ############
    # Datasets #
    ############
    def get_datasets(
        self,
        user: Optional[str] = None,
        per_page: int = 1000,
        page: int = 1,
    ) -> List[Dataset]:
        """Get a list of datasets.

        .. code-block:: python

            datasets = client.get_datasets()
            for dataset in datasets:
                print(dataset.name, dataset.description)

        Args:
            user: The user for which to get the datasets. Leave empty to get datasets of current user. Defaults to :obj:`None`.
            per_page: Pagination parameter indicating the maximum number of datasets to return. Defaults to ``1000``.
            page: Pagination parameter indicating the page to return. Defaults to ``1``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the datasets fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If one of the datasets is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        # pagination
        query_string = f"?per_page={per_page}&page={page}"

        if user is not None:
            r = self._get(f"/users/{user}/datasets/{query_string}", model=List[Dataset])
        else:
            r = self._get(f"/user/datasets/{query_string}", model=List[Dataset])

        return cast(List[Dataset], r)

    def get_dataset(self, dataset_identifier: str) -> Dataset:
        """Get a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            dataset = client.get_dataset(dataset_identifier)
            print(dataset)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the dataset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/datasets/{dataset_identifier}/", model=Dataset)

        return cast(Dataset, r)

    def add_dataset(
        self,
        name: str,
        description: str = "",
        task_type: TaskType = TaskType.SEGMENTATION_BITMAP,
        task_attributes: Optional[Union[Dict[str, Any], TaskAttributes]] = None,
        category: Category = Category.OTHER,
        public: bool = False,
        readme: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        labeling_inactivity_timeout_seconds: int = 300,
        enable_skip_labeling: bool = True,
        enable_skip_reviewing: bool = False,
        enable_ratings: bool = False,
        enable_interpolation: bool = True,
        enable_same_dimensions_track_constraint: bool = False,
        enable_save_button: bool = False,
        enable_label_status_verified: bool = False,
        enable_3d_cuboid_rotation: bool = False,
        organization: Optional[str] = None,
        enable_confirm_on_commit: bool = False,
    ) -> Dataset:
        """Add a dataset.

        .. code-block:: python

            dataset_name = 'flowers'
            description = 'A dataset containing flowers of all kinds.'
            task_type = 'segmentation-bitmap'
            dataset = client.add_dataset(dataset_name, description, task_type)
            print(dataset)

        +--------------------------------------------+---------------------------------------+
        | Task type                                  | Value                                 |
        +============================================+=======================================+
        | Image segmentation labels (bitmap)         | ``segmentation-bitmap``               |
        +--------------------------------------------+---------------------------------------+
        | Image segmentation labels (bitmap)         | ``segmentation-bitmap-highres``       |
        +--------------------------------------------+---------------------------------------+
        | Image segmentation labels (sequence)       | ``image-segmentation-sequence``       |
        +--------------------------------------------+---------------------------------------+
        | Image bounding box labels                  | ``bboxes``                            |
        +--------------------------------------------+---------------------------------------+
        | Image vector labels                        | ``vector``                            |
        +--------------------------------------------+---------------------------------------+
        | Image vector labels (sequence)             | ``image-vector-sequence``             |
        +--------------------------------------------+---------------------------------------+
        | Point cloud cuboid labels                  | ``pointcloud-cuboid``                 |
        +--------------------------------------------+---------------------------------------+
        | Point cloud cuboid labels (sequence)       | ``pointcloud-cuboid-sequence``        |
        +--------------------------------------------+---------------------------------------+
        | Point cloud segmentation labels            | ``pointcloud-segmentation``           |
        +--------------------------------------------+---------------------------------------+
        | Point cloud segmentation labels (sequence) | ``pointcloud-segmentation-sequence``  |
        +--------------------------------------------+---------------------------------------+
        | Point cloud vector labels                  | ``pointcloud-vector``                 |
        +--------------------------------------------+---------------------------------------+
        | Point cloud vector labels (sequence)       | ``pointcloud-vector-sequence``        |
        +--------------------------------------------+---------------------------------------+
        | Multisensor labels (sequence)              | ``multisensor-sequence``              |
        +--------------------------------------------+---------------------------------------+

        Args:
            name: The dataset name. Example: ``flowers``.
            description: The dataset description. Defaults to ``''``.
            task_type: The dataset's task type. Defaults to ``segmentation-bitmap``.
            task_attributes: The dataset's task attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format>`__. Defaults to ``{'format_version': '0.1', 'categories': [{'id': 1, 'name': 'object'}]}``.
            category: The dataset category. Defaults to ``other``.
            public: The dataset visibility. Defaults to :obj:`False`.
            readme: The dataset readme. Defaults to ``''``.
            metadata: Any dataset metadata. Example: ``{'day': 'sunday', 'robot_id': 3}``.
            labeling_inactivity_timeout_seconds: The number of seconds after which a user is considered inactive during labeling. Only impacts label timing metrics. Defaults to ``300``.
            enable_skip_labeling: Enable the skip button in the labeling workflow. Defaults to :obj:`True`.
            enable_skip_reviewing: Enable the skip button in the reviewing workflow. Defaults to :obj:`False`.
            enable_ratings: Enable star-ratings for labeled images. Defaults to :obj:`False`.
            enable_interpolation: Enable label interpolation in sequence datasets. Ignored for non-sequence datasets. Defaults to :obj:`True`.
            enable_same_dimensions_track_constraint: Enable constraint to keep same cuboid dimensions for the entire object track in point cloud cuboid datasets. Ignored for non-cuboid datasets. Defaults to :obj:`False`.
            enable_save_button: Enable a save button in the labeling and reviewing workflow, to save unfinished work. Defaults to :obj:`False`.
            enable_label_status_verified: Enable an additional label status "Verified". Defaults to :obj:`False`.
            enable_3d_cuboid_rotation: Enable 3D cuboid rotation (i.e., yaw, pitch and roll). Defaults to :obj:`False`.
            organization: The username of the organization for which this dataset should be created. None will create a dataset for the current user. Defaults to :obj:`None`.
            enable_confirm_on_commit: Enable a confirmation dialog when saving a sample in this dataset. Defaults to :obj:`False`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the task attributes fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of the dataset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.AlreadyExistsError`: If the dataset already exists.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if task_attributes is None:
            task_attributes = {
                "format_version": "0.1",
                "categories": [{"id": 1, "name": "object"}],
            }

        if isinstance(task_attributes, TaskAttributes):
            task_attributes = task_attributes.model_dump(mode="json", exclude_unset=True)
        else:
            try:
                task_attributes = TaskAttributes.model_validate(task_attributes).model_dump(
                    mode="json", exclude_unset=True
                )
            except pydantic.ValidationError as e:
                logger.error(
                    "Did you use the right task attributes? Please refer to the online documentation: https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format.",
                )
                raise ValidationError(message=str(e), cause=e)

        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "task_type": task_type,
            "task_attributes": task_attributes,
            "category": category,
            "public": public,
            "readme": readme,
            "labeling_inactivity_timeout_seconds": labeling_inactivity_timeout_seconds,
            "enable_skip_labeling": enable_skip_labeling,
            "enable_skip_reviewing": enable_skip_reviewing,
            "enable_ratings": enable_ratings,
            "enable_interpolation": enable_interpolation,
            "enable_same_dimensions_track_constraint": enable_same_dimensions_track_constraint,
            "enable_save_button": enable_save_button,
            "enable_label_status_verified": enable_label_status_verified,
            "enable_3d_cuboid_rotation": enable_3d_cuboid_rotation,
            "enable_confirm_on_submit": enable_confirm_on_commit,
            "data_type": "IMAGE",
        }

        if metadata:
            payload["metadata"] = metadata

        endpoint = f"/organizations/{organization}/datasets/" if organization is not None else "/user/datasets/"

        r = self._post(endpoint, data=payload, model=Dataset)

        return cast(Dataset, r)

    def update_dataset(
        self,
        dataset_identifier: str,
        description: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        task_attributes: Optional[Union[Dict[str, Any], TaskAttributes]] = None,
        category: Optional[Category] = None,
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
        """Update a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            description = 'A dataset containing flowers of all kinds.'
            dataset = client.update_dataset(dataset_identifier, description)
            print(dataset)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
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

        payload: Dict[str, Any] = {}

        if description is not None:
            payload["description"] = description

        if task_type is not None:
            payload["task_type"] = task_type

        if task_attributes is not None:
            if isinstance(task_attributes, TaskAttributes):
                task_attributes = task_attributes.model_dump(mode="json", exclude_unset=True)
            else:
                try:
                    task_attributes = TaskAttributes.model_validate(task_attributes).model_dump(
                        mode="json", exclude_unset=True
                    )
                except pydantic.ValidationError as e:
                    logger.error(
                        "Did you use the right task attributes? Please refer to the online documentation: https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format.",
                    )
                    raise ValidationError(message=str(e), cause=e)

            payload["task_attributes"] = task_attributes

        if category is not None:
            payload["category"] = category

        if public is not None:
            payload["public"] = public

        if readme is not None:
            payload["readme"] = readme

        if metadata is not None:
            payload["metadata"] = metadata

        if labeling_inactivity_timeout_seconds is not None:
            payload["labeling_inactivity_timeout_seconds"] = labeling_inactivity_timeout_seconds

        if enable_skip_labeling is not None:
            payload["enable_skip_labeling"] = enable_skip_labeling

        if enable_skip_reviewing is not None:
            payload["enable_skip_reviewing"] = enable_skip_reviewing

        if enable_ratings is not None:
            payload["enable_ratings"] = enable_ratings

        if enable_interpolation is not None:
            payload["enable_interpolation"] = enable_interpolation

        if enable_same_dimensions_track_constraint is not None:
            payload["enable_same_dimensions_track_constraint"] = enable_same_dimensions_track_constraint

        if enable_save_button is not None:
            payload["enable_save_button"] = enable_save_button

        if enable_label_status_verified is not None:
            payload["enable_label_status_verified"] = enable_label_status_verified

        if enable_3d_cuboid_rotation is not None:
            payload["enable_3d_cuboid_rotation"] = enable_3d_cuboid_rotation

        if enable_confirm_on_commit is not None:
            payload["enable_confirm_on_submit"] = enable_confirm_on_commit

        r = self._patch(f"/datasets/{dataset_identifier}/", data=payload, model=Dataset)
        # logger.info(f"Updated {dataset_identifier}")

        return cast(Dataset, r)

    def delete_dataset(self, dataset_identifier: str) -> None:
        """Delete a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            client.delete_dataset(dataset_identifier)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/datasets/{dataset_identifier}/")

    def clone_dataset(
        self,
        dataset_identifier: str,
        new_name: Optional[str] = None,
        new_task_type: Optional[TaskType] = None,
        new_public: Optional[bool] = None,
        organization: Optional[str] = None,
        clone_labels: bool = False,
    ) -> Dataset:
        """Clone a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            new_name = 'flowers-vector'
            new_task_type = 'vector'
            new_public = False
            client.clone_dataset(
                dataset_identifier,
                new_name=new_name,
                new_task_type=new_task_type,
                new_public=new_public,
            )

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            new_name: The dataset name for the clone. Defaults to ``f'{old_dataset_name}-clone'``.
            new_task_type: The task type for the clone. Defaults to the task type of the original dataset.
            new_public: The visibility for the clone. Defaults to the visibility of the original dataset.
            organization: The username of the organization for which this dataset should be created. None will create a dataset for the current user. Defaults to :obj:`None`.
            clone_labels: Whether to clone the labels of the original dataset. Defaults to :obj:`False`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the dataset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if new_name is None:
            old_name = dataset_identifier.split("/")[-1]
            new_name = f"{old_name}-clone"

        payload: Dict[str, Any] = {"name": new_name}

        if new_task_type is not None:
            payload["task_type"] = new_task_type

        if new_public is not None:
            payload["public"] = new_public

        if organization is not None:
            payload["owner"] = organization

        payload["clone_labels"] = clone_labels

        r = self._post(f"/datasets/{dataset_identifier}/clone/", data=payload, model=Dataset)

        return cast(Dataset, r)

    #################
    # Collaborators #
    #################
    def get_dataset_collaborator(self, dataset_identifier: str, username: str) -> Collaborator:
        """Get a dataset collaborator.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            username = 'john'
            client.get_dataset_collaborator(dataset_identifier, username)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            username: The username of the collaborator to be added.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the collaborator fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or dataset collaborator is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset collaborator does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        r = self._get(
            f"/datasets/{dataset_identifier}/collaborators/{username}",
            model=Collaborator,
        )

        return cast(Collaborator, r)

    def add_dataset_collaborator(
        self, dataset_identifier: str, username: str, role: Role = Role.LABELER
    ) -> Collaborator:
        """Add a dataset collaborator.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            username = 'john'
            role = 'reviewer'
            client.add_dataset_collaborator(dataset_identifier, username, role)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            username: The username of the collaborator to be added.
            role: The role of the collaborator to be added. One of ``labeler``, ``reviewer``, ``manager``, ``admin``. Defaults to ``labeler``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the collaborator fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        payload = {"user": username, "role": role}
        r = self._post(
            f"/datasets/{dataset_identifier}/collaborators/",
            data=payload,
            model=Collaborator,
        )

        return cast(Collaborator, r)

    def update_dataset_collaborator(self, dataset_identifier: str, username: str, role: Role) -> Collaborator:
        """Update a dataset collaborator.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            username = 'john'
            role = 'admin'
            client.update_dataset_collaborator(dataset_identifier, username, role)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            username: The username of the collaborator to be added.
            role: The role of the collaborator to be added. Defaults to ``labeler``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the collaborator fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or dataset collaborator is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        payload: Dict[str, Any] = {}

        if role is not None:
            payload["role"] = role

        r = self._patch(
            f"/datasets/{dataset_identifier}/collaborators/{username}",
            data=payload,
            model=Collaborator,
        )

        return cast(Collaborator, r)

    def delete_dataset_collaborator(self, dataset_identifier: str, username: str) -> None:
        """Delete a dataset collaborator.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            username = 'john'
            client.delete_dataset_collaborator(dataset_identifier, username)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            username: The username of the collaborator to be deleted.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or dataset collaborator is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        self._delete(
            f"/datasets/{dataset_identifier}/collaborators/{username}",
        )

    ###########
    # Samples #
    ###########
    def get_samples(
        self,
        dataset_identifier: str,
        labelset: Optional[str] = None,
        name: Optional[str] = None,
        label_status: Optional[Union[LabelStatus, List[LabelStatus]]] = None,
        metadata: Optional[Union[str, List[str]]] = None,
        sort: Literal["name", "created_at", "priority", "updated_at", "gt_label__updated_at"] = "name",
        direction: Literal["asc", "desc"] = "asc",
        per_page: int = 1000,
        page: int = 1,
        include_full_label: bool = False,
    ) -> List[Sample]:
        """Get the samples in a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            samples = client.get_samples(dataset_identifier)
            for sample in samples:
                print(sample.name, sample.uuid)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            labelset: If defined, this additionally returns for each sample a label summary or full label (depending on `include_full_label`) for the given labelset. Defaults to :obj:`None`.
            name: Name to filter by. Defaults to :obj:`None` (no filtering).
            label_status: Sequence of label statuses to filter by. Defaults to :obj:`None` (no filtering).
            metadata: Sequence of 'key:value' metadata attributes to filter by. Defaults to :obj:`None` (no filtering).
            sort: What to sort results by. One of ``name``, ``created_at``, ``priority``, ``updated_at``, ``gt_label__updated_at``. Defaults to ``name``.
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

        # pagination
        query_string = f"?per_page={per_page}&page={page}"

        if include_full_label and labelset is None:
            raise ValidationError(message="Please specify the `labelset` if you use `include_full_label`.")

        if labelset is not None:
            query_string += f"&labelset={labelset}"
            if include_full_label:
                query_string += "&include_full_label=1"

        # filter by name
        if name is not None:
            query_string += f"&name__contains={name}"

        # filter by metadata
        if metadata is not None:
            if isinstance(metadata, str):
                metadata = [metadata]
            query_string += f"&filters={','.join(metadata)}"

        # filter by label status
        if label_status is not None:
            if isinstance(label_status, str):
                label_status = [label_status]
            assert isinstance(label_status, list)
            # label_status = [status.upper() for status in label_status]
            query_string += f"&label_status={','.join(label_status)}"

        # sorting
        direction_str = "" if direction == "asc" else "-"
        query_string += f"&sort={direction_str}{sort}"

        r = self._get(f"/datasets/{dataset_identifier}/samples/{query_string}")
        results = r.json()

        try:
            results = TypeAdapter(List[Sample]).validate_python(results)
        except pydantic.ValidationError as e:
            raise ValidationError(message=str(e), cause=e)

        # TODO: refactor into decorator
        for s in results:
            s._inject_client(self)

        return cast(List[Sample], results)

    def get_sample(
        self,
        uuid: str,
        labelset: Optional[str] = None,
        include_signed_url: bool = False,
    ) -> Sample:
        """Get a sample.

        .. code-block:: python

            uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            sample = client.get_sample(uuid)
            print(sample)

        Args:
            uuid: The sample uuid.
            labelset: If defined, this additionally returns the label for the given labelset. Defaults to :obj:`None`.
            include_signed_url: Whether to return the pre-signed URL in case of private S3 buckets. Defaults to :obj:`False`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the sample does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        query_string = f"/samples/{uuid}/"

        if labelset is not None:
            query_string += f"?labelset={labelset}"

        if include_signed_url:
            query_string += "?include_signed_urls=1"

        r = self._get(query_string, model=Sample)

        return cast(Sample, r)

    def add_sample(
        self,
        dataset_identifier: str,
        name: str,
        attributes: Union[Dict[str, Any], SampleAttributes],
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0,
        assigned_labeler: Optional[str] = None,
        assigned_reviewer: Optional[str] = None,
        readme: str = "",
        enable_compression: bool = True,
    ) -> Sample:
        """Add a sample to a dataset.

        Note:
            - The content of the ``attributes`` field depends on the `sample type <https://docs.segments.ai/reference/sample-and-label-types/sample-types>`__.
            - If the image is on your local computer, you should first upload it to a cloud storage service like Amazon S3, Google Cloud Storage, Imgur, or to our asset storage service using :meth:`.upload_asset`.
            - If you create a sample with a URL from a public S3 bucket and you see an error on the platform, make sure to `properly configure your bucket's CORS settings <https://docs.aws.amazon.com/AmazonS3/latest/dev/cors.html>`__.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            name = 'violet.jpg'
            attributes = {
                'image': {
                    'url': 'https://example.com/violet.jpg'
                }
            }
            # Metadata and priority are optional fields.
            metadata = {
                'city': 'London',
                'weather': 'cloudy',
                'robot_id': 3
            }
            priority = 10 # Samples with higher priority value will be labeled first. Default is 0.
            sample = client.add_sample(dataset_identifier, name, attributes, metadata, priority)
            print(sample)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
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

        if isinstance(attributes, get_args(SampleAttributes)):
            attributes = attributes.model_dump(mode="json", exclude_unset=True)
        else:
            try:
                attributes = (
                    TypeAdapter(SampleAttributes)
                    .validate_python(attributes)
                    .model_dump(mode="json", exclude_unset=True)
                )
            except pydantic.ValidationError as e:
                logger.error(
                    "Did you use the right sample attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/sample-types.",
                )
                raise ValidationError(message=str(e), cause=e)

        payload: Dict[str, Any] = {
            "name": name,
            "attributes": attributes,
        }

        if metadata is not None:
            payload["metadata"] = metadata

        if priority is not None:
            payload["priority"] = priority

        if assigned_labeler is not None:
            payload["assigned_labeler"] = assigned_labeler

        if assigned_reviewer is not None:
            payload["assigned_reviewer"] = assigned_reviewer

        if readme is not None:
            payload["readme"] = readme

        r = self._post(
            f"/datasets/{dataset_identifier}/samples/", data=payload, model=Sample, gzip_compress=enable_compression
        )
        # logger.info(f"Added {name}")

        return cast(Sample, r)

    def add_samples(
        self,
        dataset_identifier: str,
        samples: List[Union[Dict[str, Any], Sample]],
        enable_compression: bool = True,
    ) -> List[Sample]:
        """Add samples to a dataset in bulk. When attempting to add samples which already exist, no error is thrown but the existing samples are returned without changes.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            samples: A list of dicts with required ``name``, ``attributes`` fields and optional ``metadata``, ``priority`` fields. See :meth:`.add_sample` for details.

        Raises:
            :exc:`KeyError`: If `name` or `attributes` is not in a sample dict.
            :exc:`~segments.exceptions.ValidationError`: If validation of the attributes of a sample fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of a sample fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.AlreadyExistsError`: If one of the samples already exists.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        for sample in samples:
            if isinstance(sample, Sample):
                sample = sample.model_dump(mode="json", exclude_unset=True)
            else:
                if "name" not in sample or "attributes" not in sample:
                    raise KeyError(f"Please add a name and attributes to your sample: {sample}")

                try:
                    sample["attributes"] = (
                        TypeAdapter(SampleAttributes)
                        .validate_python(sample["attributes"])
                        .model_dump(mode="json", exclude_unset=True)
                    )
                except pydantic.ValidationError as e:
                    logger.error(
                        "Did you use the right sample attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/sample-types.",
                    )
                    raise ValidationError(message=str(e), cause=e)

        payload = samples

        r = self._post(
            f"/datasets/{dataset_identifier}/samples_bulk/",
            data=payload,
            model=List[Sample],
            gzip_compress=enable_compression,
        )

        return cast(List[Sample], r)

    def update_sample(
        self,
        uuid: str,
        name: Optional[str] = None,
        attributes: Optional[Union[Dict[str, Any], SampleAttributes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
        assigned_labeler: Optional[str] = _NOT_ASSIGNED,
        assigned_reviewer: Optional[str] = _NOT_ASSIGNED,
        readme: Optional[str] = None,
        enable_compression: bool = True,
    ) -> Sample:
        """Update a sample.

        .. code-block:: python

            uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            metadata = {
                'city': 'London',
                'weather': 'cloudy',
                'robot_id': 3
            }
            priority = 10 # Samples with higher priority value will be labeled first. Default is 0.
            sample = client.update_sample(uuid, metadata=metadata, priority=priority)
            print(sample)

        Args:
            uuid: The sample uuid.
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

        payload: Dict[str, Any] = {}

        if name is not None:
            payload["name"] = name

        if attributes is not None:
            if isinstance(attributes, get_args(SampleAttributes)):
                attributes = attributes.model_dump(mode="json", exclude_unset=True)
            else:
                try:
                    attributes = (
                        TypeAdapter(SampleAttributes)
                        .validate_python(attributes)
                        .model_dump(mode="json", exclude_unset=True)
                    )
                except pydantic.ValidationError as e:
                    logger.error(
                        "Did you use the right sample attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/sample-types.",
                    )
                    raise ValidationError(message=str(e), cause=e)

            payload["attributes"] = attributes

        if metadata is not None:
            payload["metadata"] = metadata

        if priority is not None:
            payload["priority"] = priority

        if assigned_labeler is not _NOT_ASSIGNED:
            payload["assigned_labeler"] = assigned_labeler

        if assigned_reviewer is not _NOT_ASSIGNED:
            payload["assigned_reviewer"] = assigned_reviewer

        if readme is not None:
            payload["readme"] = readme

        r = self._patch(f"/samples/{uuid}/", data=payload, model=Sample, gzip_compress=enable_compression)
        # logger.info(f"Updated {uuid}")

        return cast(Sample, r)

    def delete_sample(self, uuid: str) -> None:
        """Delete a sample.

        .. code-block:: python

            uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            client.delete_sample(uuid)

        Args:
            uuid: The sample uuid.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/samples/{uuid}/")

    ##########
    # Labels #
    ##########
    def get_label(self, sample_uuid: str, labelset: str = "ground-truth") -> Label:
        """Get a label.

        Note:
            If the sample is unlabeled, a :exc:`~segments.exceptions.NotFoundError` will be raised.

        .. code-block:: python

            sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            label = client.get_label(sample_uuid)
            print(label)

        Args:
            sample_uuid: The sample uuid.
            labelset: The labelset this label belongs to. Defaults to ``ground-truth``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found or if the sample is unlabeled.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/labels/{sample_uuid}/{labelset}/", model=Label)

        return cast(Label, r)

    def add_label(
        self,
        sample_uuid: str,
        labelset: str,
        attributes: Union[Dict[str, Any], LabelAttributes],
        label_status: LabelStatus = LabelStatus.PRELABELED,
        score: Optional[float] = None,
        enable_compression: bool = True,
    ) -> Label:
        """Add a label to a sample.

        A label is added to a sample in relation to a `labelset`, such as the default `ground-truth` labelset, or a newly created labelset for `uploading model predictions <https://docs.segments.ai/guides/upload-model-predictions>`__. You can create a new labelset by clicking the "Add new labelset" link on the Samples tab.

        Note:
            The content of the ``attributes`` field depends on the `label type <https://docs.segments.ai/reference/sample-and-label-types/label-types>`__.

        .. code-block:: python

            sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            attributes = {
                'format_version': '0.1',
                'annotations': [
                    {
                        'id': 1,
                        'category_id': 1,
                        'type': 'bbox',
                        'points': [
                            [12.34, 56.78],
                            [90.12, 34.56]
                        ]
                    },
                ]
            }
            client.add_label(sample_uuid, attributes)

        Args:
            sample_uuid: The sample uuid.
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

        if isinstance(attributes, get_args(LabelAttributes)):
            attributes = attributes.model_dump(mode="json", exclude_unset=True)
        else:
            try:
                attributes = (
                    TypeAdapter(LabelAttributes)
                    .validate_python(attributes)
                    .model_dump(mode="json", exclude_unset=True)
                )
            except pydantic.ValidationError as e:
                logger.error(
                    "Did you use the right label attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/label-types.",
                )
                raise ValidationError(message=str(e), cause=e)

        payload: Dict[str, Any] = {
            "label_status": label_status,
            "attributes": attributes,
        }

        if score is not None:
            payload["score"] = score

        r = self._put(
            f"/labels/{sample_uuid}/{labelset}/", data=payload, model=Label, gzip_compress=enable_compression
        )

        return cast(Label, r)

    def update_label(
        self,
        sample_uuid: str,
        labelset: str,
        attributes: Optional[Union[Dict[str, Any], LabelAttributes]] = None,
        label_status: Optional[LabelStatus] = None,
        score: Optional[float] = None,
        enable_compression: bool = True,
    ) -> Label:
        """Update a label.

        .. code-block:: python

            sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            attributes = {
                'format_version': '0.1',
                'annotations': [
                    {
                        'id': 1,
                        'category_id': 1,
                        'type': 'bbox',
                        'points': [
                            [12.34, 56.78],
                            [90.12, 34.56]
                        ]
                    },
                ]
            }
            client.update_label(sample_uuid, attributes)

        Args:
            sample_uuid: The sample uuid.
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

        payload: Dict[str, Any] = {}

        if attributes is not None:
            if isinstance(attributes, get_args(LabelAttributes)):
                attributes = attributes.model_dump(mode="json", exclude_unset=True)
            else:
                try:
                    attributes = (
                        TypeAdapter(LabelAttributes)
                        .validate_python(attributes)
                        .model_dump(mode="json", exclude_unset=True)
                    )
                except pydantic.ValidationError as e:
                    logger.error(
                        "Did you use the right label attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/label-types.",
                    )
                    raise ValidationError(message=str(e), cause=e)

            payload["attributes"] = attributes

        if label_status is not None:
            payload["label_status"] = label_status

        if score is not None:
            payload["score"] = score

        r = self._patch(
            f"/labels/{sample_uuid}/{labelset}/", data=payload, model=Label, gzip_compress=enable_compression
        )

        return cast(Label, r)

    def delete_label(self, sample_uuid: str, labelset: str) -> None:
        """Delete a label.

        .. code-block:: python

            sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            labelset = 'ground-truth'
            client.delete_label(sample_uuid, labelset)

        Args:
            sample_uuid: The sample uuid.
            labelset: The labelset this label belongs to.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/labels/{sample_uuid}/{labelset}/")

    #############
    # Labelsets #
    #############
    def get_labelsets(self, dataset_identifier: str) -> List[Labelset]:
        """Get the labelsets in a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            labelsets = client.get_labelsets(dataset_identifier)
            for labelset in labelsets:
                print(labelset.name, labelset.description)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the labelsets fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/datasets/{dataset_identifier}/labelsets/", model=List[Labelset])

        return cast(List[Labelset], r)

    def get_labelset(self, dataset_identifier: str, name: str) -> Labelset:
        """Get a labelset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            name = 'model-predictions'
            labelset = client.get_labelset(dataset_identifier, name)
            print(labelset)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the labelset.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the labelset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/datasets/{dataset_identifier}/labelsets/{name}/", model=Labelset)

        return cast(Labelset, r)

    def add_labelset(self, dataset_identifier: str, name: str, description: str = "") -> Labelset:
        """Add a labelset to a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            name = 'model-predictions-resnet50'
            client.add_labelset(dataset_identifier, name)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
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

        payload = {
            "name": name,
            "description": description,
            "attributes": "{}",
        }
        r = self._post(
            f"/datasets/{dataset_identifier}/labelsets/",
            data=payload,
            model=Labelset,
        )

        return cast(Labelset, r)

    def delete_labelset(self, dataset_identifier: str, name: str) -> None:
        """Delete a labelset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            name = 'model-predictions'
            client.delete_labelset(dataset_identifier, name)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the labelset.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/datasets/{dataset_identifier}/labelsets/{name}/")

    ##########
    # Issues #
    ##########
    def get_issues(self, dataset_identifier: str) -> List[Issue]:
        """Get all issues for a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            issues = client.get_issues(dataset_identifier)
            for issue in issues:
                print(issue.uuid, issue.description, issue.status, issue.sample_uuid)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/datasets/{dataset_identifier}/issues/", model=List[Issue])

        return cast(List[Issue], r)

    def add_issue(
        self,
        sample_uuid: str,
        description: str,
        status: IssueStatus = IssueStatus.OPEN,
    ) -> Issue:
        """Add an issue to a sample.

        .. code-block:: python

            sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            description = 'You forgot to label the cars in this image.'

            client.add_issue(sample_uuid, description)

        Args:
            sample_uuid: The sample uuid.
            description: The issue description.
            status: The issue status. One of ``OPEN`` or ``CLOSED``. Defaults to ``OPEN``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the issue fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        payload: Dict[str, Any] = {
            "description": description,
            "status": status,
        }

        r = self._post(f"/samples/{sample_uuid}/issues/", data=payload, model=Issue)

        return cast(Issue, r)

    def update_issue(
        self,
        uuid: str,
        description: Optional[str] = None,
        status: Optional[IssueStatus] = None,
    ) -> Issue:
        """Add an issue to a sample.

        .. code-block:: python

            uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            description = 'You forgot to label the cars in this image.'

            client.update_issue(sample_uuid, description)

        Args:
            uuid: The issue uuid.
            description: The issue description. Defaults to :obj:`None`.
            status: The issue status. One of ``OPEN`` or ``CLOSED``. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the issue fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the issue is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        payload: Dict[str, Any] = {}

        if description is not None:
            payload["description"] = description

        if status is not None:
            payload["status"] = status

        r = self._patch(f"/issues/{uuid}/", data=payload, model=Issue)

        return cast(Issue, r)

    def delete_issue(self, uuid: str) -> None:
        """Delete an issue.

        .. code-block:: python

            uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            client.delete_issue(uuid)

        Args:
            uuid: The issue uuid.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the issue is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/issues/{uuid}/")

    ############
    # Releases #
    ############
    def get_releases(self, dataset_identifier: str) -> List[Release]:
        """Get the releases in a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            releases = client.get_releases(dataset_identifier)
            for release in releases:
                print(release.name, release.description, release.attributes.url)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the releases fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/datasets/{dataset_identifier}/releases/", model=List[Release])

        return cast(List[Release], r)

    def get_release(self, dataset_identifier: str, name: str) -> Release:
        """Get a release.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            name = 'v0.1'
            release = client.get_release(dataset_identifier, name)
            print(release)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the release.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the release fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or release is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the dataset does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/datasets/{dataset_identifier}/releases/{name}/", model=Release)

        return cast(Release, r)

    def add_release(self, dataset_identifier: str, name: str, description: str = "") -> Release:
        """Add a release to a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            name = 'v0.1'
            description = 'My first release.'
            release = client.add_release(dataset_identifier, name, description)
            print(release)

            # Wait for the release to be created
            while release.status == ReleaseStatus.PENDING:
                release = client.get_release(dataset_identifier, name)
                sleep(5)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
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

        payload = {"name": name, "description": description}
        r = self._post(
            f"/datasets/{dataset_identifier}/releases/",
            data=payload,
            model=Release,
        )

        return cast(Release, r)

    def delete_release(self, dataset_identifier: str, name: str) -> None:
        """Delete a release.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            name = 'v0.1'
            client.delete_release(dataset_identifier, name)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the release.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset or release is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/datasets/{dataset_identifier}/releases/{name}/")

    #############
    # Workunits #
    #############
    def get_workunits(
        self,
        dataset_identifier: str,
        sort: Literal["created_at"] = "created_at",
        direction: Literal["asc", "desc"] = "desc",
        start: Optional[str] = None,
        end: Optional[str] = None,
        per_page: int = 1000,
        page: int = 1,
    ) -> List[Workunit]:
        """Get the workunits in a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            workunits = client.get_workunits(dataset_identifier)
            for workunit in workunits:
                print(workunit.created_at)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
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

        # pagination
        query_string = f"?per_page={per_page}&page={page}"

        # sorting
        direction_str = "" if direction == "asc" else "-"
        query_string += f"&sort={direction_str}{sort}"

        # filter by start datetime
        if start is not None:
            query_string += f"&start={start}"

        # filter by end datetime
        if end is not None:
            query_string += f"&end={end}"

        r = self._get(f"/datasets/{dataset_identifier}/workunits/{query_string}")
        results = r.json()

        try:
            results = TypeAdapter(List[Workunit]).validate_python(results)
        except pydantic.ValidationError as e:
            raise ValidationError(message=str(e), cause=e)

        return cast(List[Workunit], results)

    ##########
    # Assets #
    ##########
    def upload_asset(self, file: Union[TextIO, BinaryIO], filename: str = "label.png") -> File:
        """Upload an asset.

        .. code-block:: python

            filename = '/home/jane/flowers/violet.jpg'
            with open(filename, 'rb') as f:
                filename = 'violet.jpg'
                asset = client.upload_asset(f, filename)
            image_url = asset.url
            print(image_url)

        Args:
            file: A file object.
            filename: The file name. Defaults to ``label.png``.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the file fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._post("/assets/", data={"filename": filename})
        presigned_post_fields = PresignedPostFields.model_validate(r.json()["presignedPostFields"])
        self._upload_to_aws(file, presigned_post_fields.url, presigned_post_fields.fields)

        try:
            f = File.model_validate(r.json())
        except pydantic.ValidationError as e:
            raise ValidationError(message=str(e), cause=e)

        return f

    ####################
    # Helper functions #
    ####################
    @convert_model
    @handle_exceptions
    def _get(
        self,
        endpoint: str,
        auth: bool = True,
        model: Optional[T] = None,
    ) -> requests.Response:
        """Send a GET request.

        Args:
            endpoint: The API endpoint.
            auth: If we want to authorize the request. Defaults to :obj:`True`.
            model: The class to parse the JSON response into. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """

        headers = self._get_headers(auth)

        r = self.api_session.get(urllib.parse.urljoin(self.api_url, endpoint), headers=headers)

        return r

    @convert_model
    @handle_exceptions
    def _post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        model: Optional[T] = None,
        gzip_compress: bool = False,
    ) -> requests.Response:
        """Send a POST request.

        Args:
            endpoint: The API endpoint.
            data: The JSON data. Defaults to :obj:`None`.
            auth: If we want to authorize the request with the API key. Defaults to :obj:`True`.
            model: The class to parse the JSON response into. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """
        headers = self._get_headers(auth)

        request_args = {"url": urllib.parse.urljoin(self.api_url, endpoint), "headers": headers}
        self._set_request_body(data, request_args, gzip_compress)

        r = self.api_session.post(**request_args)

        return r

    @convert_model
    @handle_exceptions
    def _put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        model: Optional[T] = None,
        gzip_compress: bool = False,
    ) -> requests.Response:
        """Send a PUT request.

        Args:
            endpoint: The API endpoint.
            data: The JSON data. Defaults to :obj:`None`.
            auth: If we want to authorize the request with the API key. Defaults to :obj:`True`.
            model: The class to parse the JSON response into. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """
        headers = self._get_headers(auth)

        request_args = {"url": urllib.parse.urljoin(self.api_url, endpoint), "headers": headers}
        self._set_request_body(data, request_args, gzip_compress)
        r = self.api_session.put(**request_args)

        return r

    @convert_model
    @handle_exceptions
    def _patch(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        model: Optional[T] = None,
        gzip_compress: bool = False,
    ) -> requests.Response:
        """Send a PATCH request.

        Args:
            endpoint: The API endpoint.
            data: The JSON data. Defaults to :obj:`None`.
            auth: If we want to authorize the request with the API key. Defaults to :obj:`True`.
            model: The class to parse the JSON response into. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """
        headers = self._get_headers(auth)
        request_args = {"url": urllib.parse.urljoin(self.api_url, endpoint), "headers": headers}
        self._set_request_body(data, request_args, gzip_compress)

        r = self.api_session.patch(**request_args)

        return r

    @convert_model
    @handle_exceptions
    def _delete(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        model: Optional[T] = None,
    ) -> requests.Response:
        """Send a DELETE request.

        Args:
            endpoint: The API endpoint.
            data: The JSON data. Defaults to :obj:`None`.
            auth: If we want to authorize the request with the API key. Defaults to :obj:`True`.
            model: The class to parse the JSON response into. Defaults to :obj:`None`.

        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """
        headers = self._get_headers(auth)

        r = self.api_session.delete(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )

        return r

    def _get_headers(self, auth: bool = True) -> Dict[str, str]:
        """Get the authorization header with the API key."""
        headers = {
            "X-source": "python-sdk",
            "Segments-SDK-Version": VERSION,
        }
        if auth and self.api_key:
            headers["Authorization"] = f"APIKey {self.api_key}"
        return headers

    def _set_request_body(self, data: Dict[str, Any], request_args: Dict[str, Any], gzip_compress: bool):
        """Set the appropriate request body and headers based on whether gzip compression is requested."""
        if gzip_compress:
            request_args["data"] = gzip.compress(json.dumps(data).encode())
            request_args["headers"]["Content-Encoding"] = "gzip"
            request_args["headers"]["Content-Type"] = "application/json"
            request_args["headers"]["Content-Length"] = str(len(request_args["data"]))
        else:
            request_args["json"] = data

    @handle_exceptions
    def _upload_to_aws(self, file: Union[TextIO, BinaryIO], url: str, aws_fields: AWSFields) -> requests.Response:
        """Upload file to AWS.

        Args:
            file: The file we want to upload.
            url: The request's url.
            aws_fields: The AWS fields.

        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        files = {"file": file}
        r = self.s3_session.post(url, files=files, data=aws_fields)

        return r
