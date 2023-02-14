# https://adamj.eu/tech/2021/05/13/python-type-hints-how-to-fix-circular-imports/
from __future__ import annotations

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

import numpy.typing as npt
import pydantic
import requests
from pydantic import parse_obj_as
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
    AuthHeader,
    AWSFields,
    Category,
    Collaborator,
    Dataset,
    File,
    Issue,
    IssueStatus,
    Label,
    LabelAttributes,
    Labelset,
    LabelStatus,
    PresignedPostFields,
    Release,
    Role,
    Sample,
    SampleAttributes,
    TaskAttributes,
    TaskType,
)
from typing_extensions import Literal, get_args

################################
# Constants and type variables #
################################
logger = logging.getLogger(__name__)
T = TypeVar("T")
VERSION = "1.0.19"


####################
# Helper functions #
####################
# Error handling: https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
def handle_exceptions(
    f: Callable[..., requests.Response]
) -> Callable[..., Union[requests.Response, T]]:
    """Catch exceptions and throw Segments exceptions.

    Args:
        model: The class to parse the JSON response into. Defaults to :obj:`None`.
    Returns:
        A wrapper function (of this exception handler decorator).
    Raises:
        :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
        :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
        :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
        :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
    """

    def throw_segments_exception(
        *args: Any,
        model: Optional[Type[T]] = None,
        **kwargs: Any,
    ) -> Union[requests.Response, T]:
        try:
            r = f(*args, **kwargs)
            r.raise_for_status()
            if r.content:
                r_json = r.json()
                # Check if the API limit is exceeded
                if isinstance(r_json, dict):
                    message = r_json.get("detail", "")
                    if "throttled" in message:
                        raise APILimitError(message)
                if model:
                    m = parse_obj_as(model, r_json)
                    return m
            return r
        except requests.exceptions.Timeout as e:
            # Maybe set up for a retry, or continue in a retry loop
            raise TimeoutError(message=str(e), cause=e)
        except requests.exceptions.HTTPError as e:
            # Make string comparison case insensitive.
            text = e.response.text.lower()
            if "not found" in text or "does not exist" in text:
                raise NotFoundError(message=text, cause=e)
            if "already exists" in text or "already have" in text:
                raise AlreadyExistsError(message=text, cause=e)
            if (
                "cannot be added as collaborator" in text
                or "is already a collaborator" in text
            ):
                raise CollaboratorError(message=text, cause=e)
            if (
                "cannot leave the organization" in text
                or "need to be an administrator" in text
                or "do not have permission" in text
            ):
                raise AuthorizationError(message=text, cause=e)
            if "free trial ended" in text or "exceeded user limit" in text:
                raise SubscriptionError(message=text, cause=e)
            raise NetworkError(message=text, cause=e)
        except requests.exceptions.TooManyRedirects as e:
            # Tell the user their URL was bad and try a different one
            raise NetworkError(message="Bad url, please try a different one.", cause=e)
        except requests.exceptions.RequestException as e:
            logger.error(f"Unknown error: {e}")
            raise NetworkError(message=str(e), cause=e)
        except pydantic.ValidationError as e:
            raise ValidationError(message=str(e), cause=e)

    return throw_segments_exception


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
                    message="Please provide the api_key argument or set SEGMENTS_API_KEY in your environment."
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

        try:
            r = self._get(f"/api_status/?lib_version={VERSION}")
            if r.status_code == 200:
                logger.info("Initialized successfully.")
        except NetworkError as e:
            if (
                cast(requests.exceptions.RequestException, e.cause).response.status_code
                == 426
            ):
                logger.warning(
                    "There's a new version available. Please upgrade by running 'pip install --upgrade segments-ai'"
                )
            else:
                raise AuthenticationError(
                    message="Something went wrong. Did you use the right API key?"
                )

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

    ############
    # Datasets #
    ############
    def get_datasets(self, user: Optional[str] = None) -> List[Dataset]:
        """Get a list of datasets.

        .. code-block:: python

            datasets = client.get_datasets()
            for dataset in datasets:
                print(dataset.name, dataset.description)

        Args:
            user: The user for which to get the datasets. Leave empty to get datasets of current user. Defaults to :obj:`None`.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the datasets fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If one of the datasets is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if user:
            r = self._get(f"/users/{user}/datasets/", model=List[Dataset])
        else:
            r = self._get("/user/datasets/", model=List[Dataset])

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
        task_type: TaskType = "segmentation-bitmap",
        task_attributes: Optional[Union[Dict[str, Any], TaskAttributes]] = None,
        category: Category = "other",
        public: bool = False,
        readme: str = "",
        enable_skip_labeling: bool = True,
        enable_skip_reviewing: bool = False,
        enable_ratings: bool = False,
        enable_interpolation: bool = True,
        enable_same_dimensions_track_constraint: bool = False,
        enable_save_button: bool = False,
        enable_label_status_verified: bool = False,
        organization: Optional[str] = None,
    ) -> Dataset:
        """Add a dataset.

        .. code-block:: python

            dataset_name = 'flowers'
            description = 'A dataset containing flowers of all kinds.'
            task_type = 'segmentation-bitmap'
            dataset = client.add_dataset(dataset_name, description, task_type)
            print(dataset)

        +-------------------------------------------+---------------------------------------+
        | Task type                                 | Value                                 |
        +===========================================+=======================================+
        | Image segmentation labels (bitmap)        | ``segmentation-bitmap``               |
        +-------------------------------------------+---------------------------------------+
        | Image bounding box labels                 | ``bboxes``                            |
        +-------------------------------------------+---------------------------------------+
        | Image vector labels                       | ``vector``                            |
        +-------------------------------------------+---------------------------------------+
        | Pointcloud cuboid labels                  | ``pointcloud-cuboid``                 |
        +-------------------------------------------+---------------------------------------+
        | Pointcloud cuboid labels (sequence)       | ``pointcloud-cuboid-sequence``        |
        +-------------------------------------------+---------------------------------------+
        | Pointcloud segmentation labels            | ``pointcloud-segmentation``           |
        +-------------------------------------------+---------------------------------------+
        | Pointcloud segmentation labels (sequence) | ``pointcloud-segmentation-sequence``  |
        +-------------------------------------------+---------------------------------------+
        | Text named entity labels                  | ``text-named-entities``               |
        +-------------------------------------------+---------------------------------------+
        | Text span categorization labels           | ``text-span-categorization``          |
        +-------------------------------------------+---------------------------------------+

        Args:
            name: The dataset name. Example: ``flowers``.
            description: The dataset description. Defaults to ``''``.
            task_type: The dataset's task type. Defaults to ``segmentation-bitmap``.
            task_attributes: The dataset's task attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format>`__. Defaults to ``{'format_version': '0.1', 'categories': [{'id': 1, 'name': 'object'}]}``.
            category: The dataset category. Defaults to ``other``.
            public: The dataset visibility. Defaults to :obj:`False`.
            readme: The dataset readme. Defaults to ``''``.
            enable_skip_labeling: Enable the skip button in the labeling workflow. Defaults to :obj:`True`.
            enable_skip_reviewing: Enable the skip button in the reviewing workflow. Defaults to :obj:`False`.
            enable_ratings: Enable star-ratings for labeled images. Defaults to :obj:`False`.
            enable_interpolation: Enable label interpolation in sequence datasets. Ignored for non-sequence datasets. Defaults to :obj:`True`.
            enable_same_dimensions_track_constraint: Enable constraint to keep same cuboid dimensions for the entire object track in point cloud cuboid datasets. Ignored for non-cuboid datasets. Defaults to :obj:`False`.
            enable_save_button: Enable a save button in the labeling and reviewing workflow, to save unfinished work. Defaults to :obj:`False`.
            enable_label_status_verified: Enable an additional label status "Verified". Defaults to :obj:`False`.
            organization: The username of the organization for which this dataset should be created. None will create a dataset for the current user. Defaults to :obj:`None`.
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

        if type(task_attributes) is dict:
            try:
                TaskAttributes.parse_obj(task_attributes)
            except pydantic.ValidationError as e:
                logger.error(
                    "Did you use the right task attributes? Please refer to the online documentation: https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format.",
                )
                raise ValidationError(message=str(e), cause=e)
        elif type(task_attributes) is TaskAttributes:
            task_attributes = task_attributes.dict()

        payload: Dict[str, Any] = {
            "name": name,
            "description": description,
            "task_type": task_type,
            "task_attributes": task_attributes,
            "category": category,
            "public": public,
            "readme": readme,
            "enable_skip_labeling": enable_skip_labeling,
            "enable_skip_reviewing": enable_skip_reviewing,
            "enable_ratings": enable_ratings,
            "enable_interpolation": enable_interpolation,
            "enable_same_dimensions_track_constraint": enable_same_dimensions_track_constraint,
            "enable_save_button": enable_save_button,
            "enable_label_status_verified": enable_label_status_verified,
            "data_type": "IMAGE",
        }

        endpoint = (
            f"/organizations/{organization}/datasets/"
            if organization is not None
            else "/user/datasets/"
        )

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
        enable_skip_labeling: Optional[bool] = None,
        enable_skip_reviewing: Optional[bool] = None,
        enable_ratings: Optional[bool] = None,
        enable_interpolation: Optional[bool] = None,
        enable_same_dimensions_track_constraint: Optional[bool] = None,
        enable_save_button: Optional[bool] = None,
        enable_label_status_verified: Optional[bool] = None,
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
            enable_skip_labeling: Enable the skip button in the labeling workflow. Defaults to :obj:`None`.
            enable_skip_reviewing: Enable the skip button in the reviewing workflow. Defaults to :obj:`None`.
            enable_ratings: Enable star-ratings for labeled images. Defaults to :obj:`None`.
            enable_interpolation: Enable label interpolation in sequence datasets. Ignored for non-sequence datasets. Defaults to :obj:`None`.
            enable_same_dimensions_track_constraint: Enable constraint to keep same cuboid dimensions for the entire object track in point cloud cuboid datasets. Ignored for non-cuboid datasets. Defaults to :obj:`None`.
            enable_save_button: Enable a save button in the labeling and reviewing workflow, to save unfinished work. Defaults to :obj:`False`.
            enable_label_status_verified: Enable an additional label status "Verified". Defaults to :obj:`False`.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the dataset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        payload: Dict[str, Any] = {}

        if description:
            payload["description"] = description

        if task_type:
            payload["task_type"] = task_type

        if task_attributes:
            payload["task_attributes"] = (
                task_attributes.dict()
                if type(task_attributes) is TaskAttributes
                else task_attributes
            )

        if category:
            payload["category"] = category

        if public:
            payload["public"] = public

        if readme:
            payload["readme"] = readme

        if enable_skip_labeling:
            payload["enable_skip_labeling"] = enable_skip_labeling

        if enable_skip_reviewing:
            payload["enable_skip_reviewing"] = enable_skip_reviewing

        if enable_ratings:
            payload["enable_ratings"] = enable_ratings

        if enable_interpolation:
            payload["enable_interpolation"] = enable_interpolation

        if enable_same_dimensions_track_constraint:
            payload[
                "enable_same_dimensions_track_constraint"
            ] = enable_same_dimensions_track_constraint

        if enable_save_button:
            payload["enable_save_button"] = enable_save_button

        if enable_label_status_verified:
            payload["enable_label_status_verified"] = enable_label_status_verified

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

        if new_task_type:
            payload["task_type"] = new_task_type

        if new_public:
            payload["public"] = new_public

        r = self._post(
            f"/datasets/{dataset_identifier}/clone/", data=payload, model=Dataset
        )

        return cast(Dataset, r)

    #################
    # Collaborators #
    #################
    def get_dataset_collaborator(
        self, dataset_identifier: str, username: str
    ) -> Collaborator:
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
        self, dataset_identifier: str, username: str, role: Role = "labeler"
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
            role: The role of the collaborator to be added. Defaults to ``labeler``.
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

    def update_dataset_collaborator(
        self, dataset_identifier: str, username: str, role: Role
    ) -> Collaborator:
        """ "Update a dataset collaborator.

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
        payload = {"role": role}
        r = self._patch(
            f"/datasets/{dataset_identifier}/collaborators/{username}",
            data=payload,
            model=Collaborator,
        )

        return cast(Collaborator, r)

    def delete_dataset_collaborator(
        self, dataset_identifier: str, username: str
    ) -> None:
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
        name: Optional[str] = None,
        label_status: Optional[Union[LabelStatus, List[LabelStatus]]] = None,
        metadata: Optional[Union[str, List[str]]] = None,
        sort: Literal["name", "created", "priority"] = "name",
        direction: Literal["asc", "desc"] = "asc",
        per_page: int = 1000,
        page: int = 1,
    ) -> List[Sample]:
        """Get the samples in a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            samples = client.get_samples(dataset_identifier)
            for sample in samples:
                print(sample.name, sample.uuid)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: Name to filter by. Defaults to :obj:`None` (no filtering).
            label_status: Sequence of label statuses to filter by. Defaults to :obj:`None` (no filtering).
            metadata: Sequence of 'key:value' metadata attributes to filter by. Defaults to :obj:`None` (no filtering).
            sort: What to sort results by. One of ``name``, ``created``, ``priority``. Defaults to ``name``.
            direction: Sorting direction. One of ``asc`` (ascending) or ``desc`` (descending). Defaults to ``asc``.
            per_page: Pagination parameter indicating the maximum number of samples to return. Defaults to ``1000``.
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

        # filter by name
        if name:
            query_string += f"&name__contains={name}"

        # filter by metadata
        if metadata:
            if isinstance(metadata, str):
                metadata = [metadata]
            query_string += f"&filters={','.join(metadata)}"

        # filter by label status
        if label_status:
            if isinstance(label_status, str):
                label_status = [label_status]
            assert isinstance(label_status, list)
            # label_status = [status.upper() for status in label_status]
            query_string += "&labelset=ground-truth&label_status={}".format(
                ",".join(label_status)
            )

        # sorting
        sort_dict = {"name": "name", "created": "created_at", "priority": "priority"}
        if sort != "name":
            direction_str = "" if direction == "asc" else "-"
            sort_str = sort_dict[sort]
            query_string += f"&sort={direction_str}{sort_str}"

        r = self._get(f"/datasets/{dataset_identifier}/samples/{query_string}")
        results = r.json()

        # TODO
        for result in results:
            result.pop("label", None)

        try:
            results = parse_obj_as(List[Sample], results)
        except pydantic.ValidationError as e:
            raise ValidationError(message=str(e), cause=e)

        return cast(List[Sample], results)

    def get_sample(self, uuid: str, labelset: Optional[str] = None) -> Sample:
        """Get a sample.

        .. code-block:: python

            uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
            sample = client.get_sample(uuid)
            print(sample)

        Args:
            uuid: The sample uuid.
            labelset: If defined, this additionally returns the label for the given labelset. Defaults to :obj:`None`.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid (e.g., if the sample does not exist) or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        query_string = f"/samples/{uuid}/"

        if labelset:
            query_string += f"?labelset={labelset}"

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
        embedding: Optional[Union[npt.NDArray[Any], List[float]]] = None,
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
            embedding: Embedding of this sample represented by an array of floats.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the sample attributes fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of the sample fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.AlreadyExistsError`: If the sample already exists.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if type(attributes) is dict:
            try:
                parse_obj_as(SampleAttributes, attributes)
            except pydantic.ValidationError as e:
                logger.error(
                    "Did you use the right sample attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/sample-types.",
                )
                raise ValidationError(message=str(e), cause=e)
        elif type(attributes) in get_args(SampleAttributes):
            attributes = attributes.dict()

        payload: Dict[str, Any] = {
            "name": name,
            "attributes": attributes,
        }

        if metadata:
            payload["metadata"] = metadata

        if priority:
            payload["priority"] = priority

        if assigned_labeler:
            payload["assigned_labeler"] = assigned_labeler

        if assigned_reviewer:
            payload["assigned_reviewer"] = assigned_reviewer

        if embedding:
            payload["embedding"] = embedding

        r = self._post(
            f"/datasets/{dataset_identifier}/samples/",
            data=payload,
            model=Sample,
        )
        # logger.info(f"Added {name}")

        return cast(Sample, r)

    def add_samples(
        self, dataset_identifier: str, samples: List[Union[Dict[str, Any], Sample]]
    ) -> List[Sample]:
        """Add samples to a dataset in bulk. When attempting to add samples which already exist, no error is thrown but the existing samples are returned without changes.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            samples: A list of dicts with required ``name``, ``attributes`` fields and optional ``metadata``, ``priority``, ``embedding`` fields. See :meth:`.add_sample` for details.
        Raises:
            :exc:`KeyError`: If 'name' or 'attributes' is not in a sample dict.
            :exc:`~segments.exceptions.ValidationError`: If validation of the attributes of a sample fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of a sample fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the dataset is not found.
            :exc:`~segments.exceptions.AlreadyExistsError`: If one of the samples already exists.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        # Check the input
        for sample in samples:
            if type(sample) is dict:
                if "name" not in sample or "attributes" not in sample:
                    raise KeyError(
                        f"Please add a name and attributes to your sample: {sample}"
                    )

                try:
                    parse_obj_as(SampleAttributes, sample["attributes"])
                except pydantic.ValidationError as e:
                    logger.error(
                        "Did you use the right sample attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/sample-types.",
                    )
                    raise ValidationError(message=str(e), cause=e)
            elif type(sample) is Sample:
                sample = sample.dict()

        payload = samples

        r = self._post(
            f"/datasets/{dataset_identifier}/samples_bulk/",
            data=payload,
            model=List[Sample],
        )

        return cast(List[Sample], r)

    def update_sample(
        self,
        uuid: str,
        name: Optional[str] = None,
        attributes: Optional[Union[Dict[str, Any], SampleAttributes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: Optional[float] = None,
        assigned_labeler: Optional[str] = None,
        assigned_reviewer: Optional[str] = None,
        embedding: Optional[Union[npt.NDArray[Any], List[float]]] = None,
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
            embedding: Embedding of this sample represented by list of floats.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.NotFoundError`: If the sample is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        payload: Dict[str, Any] = {}

        if name:
            payload["name"] = name

        if attributes:
            payload["attributes"] = (
                attributes.dict()
                if type(attributes) in get_args(SampleAttributes)
                else attributes
            )

        if metadata:
            payload["metadata"] = metadata

        if priority:
            payload["priority"] = priority

        if assigned_labeler:
            payload["assigned_labeler"] = assigned_labeler

        if assigned_reviewer:
            payload["assigned_reviewer"] = assigned_reviewer

        if embedding:
            payload["embedding"] = embedding

        r = self._patch(f"/samples/{uuid}/", data=payload, model=Sample)
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
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
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
        label_status: LabelStatus = "PRELABELED",
        score: Optional[float] = None,
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
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the attributes fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if type(attributes) is dict:
            try:
                parse_obj_as(LabelAttributes, attributes)
            except pydantic.ValidationError as e:
                logger.error(
                    "Did you use the right label attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/label-types.",
                )
                raise ValidationError(message=str(e), cause=e)
        elif type(attributes) in get_args(LabelAttributes):
            attributes = attributes.dict()

        payload: Dict[str, Any] = {
            "label_status": label_status,
            "attributes": attributes,
        }

        if score:
            payload["score"] = score

        r = self._put(f"/labels/{sample_uuid}/{labelset}/", data=payload, model=Label)

        return cast(Label, r)

    def update_label(
        self,
        sample_uuid: str,
        labelset: str,
        attributes: Union[Dict[str, Any], LabelAttributes],
        label_status: LabelStatus = "PRELABELED",
        score: Optional[float] = None,
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
            attributes: The label attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/label-types>`__.
            label_status: The label status. Defaults to ``PRELABELED``.
            score: The label score. Defaults to :obj:`None`.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NotFoundError`: If the sample or labelset is not found.
            :exc:`~segments.exceptions.NetworkError`: If the request is not valid or if the server experienced an error.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        payload: Dict[str, Any] = {}

        if attributes:
            payload["attributes"] = (
                attributes.dict()
                if type(attributes) in get_args(LabelAttributes)
                else attributes
            )

        if label_status:
            payload["label_status"] = label_status

        if score:
            payload["score"] = score

        r = self._patch(f"/labels/{sample_uuid}/{labelset}/", data=payload, model=Label)

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

        r = self._get(
            f"/datasets/{dataset_identifier}/labelsets/", model=List[Labelset]
        )

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

        r = self._get(
            f"/datasets/{dataset_identifier}/labelsets/{name}/", model=Labelset
        )

        return cast(Labelset, r)

    def add_labelset(
        self, dataset_identifier: str, name: str, description: str = ""
    ) -> Labelset:
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
        status: IssueStatus = "OPEN",
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

        if description:
            payload["description"] = description

        if status:
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

    def add_release(
        self, dataset_identifier: str, name: str, description: str = ""
    ) -> Release:
        """Add a release to a dataset.

        .. code-block:: python

            dataset_identifier = 'jane/flowers'
            name = 'v0.1'
            description = 'My first release.'
            release = client.add_release(dataset_identifier, name, description)
            print(release)

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

    ##########
    # Assets #
    ##########
    def upload_asset(
        self, file: Union[TextIO, BinaryIO], filename: str = "label.png"
    ) -> File:
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
        presigned_post_fields = PresignedPostFields.parse_obj(
            r.json()["presignedPostFields"]
        )
        self._upload_to_aws(
            file, presigned_post_fields.url, presigned_post_fields.fields
        )

        try:
            f = File.parse_obj(r.json())
        except pydantic.ValidationError as e:
            raise ValidationError(message=str(e), cause=e)

        return f

    ####################
    # Helper functions #
    ####################
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

        headers = self._get_auth_header() if auth else None

        r = self.api_session.get(
            urllib.parse.urljoin(self.api_url, endpoint), headers=headers
        )

        return r

    @handle_exceptions
    def _post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        model: Optional[T] = None,
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
        headers = self._get_auth_header() if auth else None

        r = self.api_session.post(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )

        return r

    @handle_exceptions
    def _put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        model: Optional[T] = None,
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
        headers = self._get_auth_header() if auth else None

        r = self.api_session.put(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )

        return r

    @handle_exceptions
    def _patch(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        model: Optional[T] = None,
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
        headers = self._get_auth_header() if auth else None

        r = self.api_session.patch(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )

        return r

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
        headers = self._get_auth_header() if auth else None

        r = self.api_session.delete(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )

        return r

    def _get_auth_header(self) -> Optional[AuthHeader]:
        """Get the authorization header with the API key."""
        return {"Authorization": f"APIKey {self.api_key}"} if self.api_key else None

    @handle_exceptions
    def _upload_to_aws(
        self, file: Union[TextIO, BinaryIO], url: str, aws_fields: AWSFields
    ) -> requests.Response:
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
