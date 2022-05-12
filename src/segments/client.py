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
    APILimitError,
    AuthenticationError,
    NetworkError,
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
from typing_extensions import Literal

#############
# Variables #
#############
logger = logging.getLogger(__name__)
T = TypeVar("T")


####################
# Helper functions #
####################
# Error handling: https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
def exception_handler(
    f: Callable[..., requests.Response]
) -> Callable[..., Union[requests.Response, T]]:
    """Catch exceptions and throw Segments exceptions.

    Args:
        pydantic_model: The pydantic class to parse the JSON response into. Defaults to :obj:`None`.
    Returns:
        A wrapper function (of this exception handler decorator).
    Raises:
        :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
        :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
        :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error) - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
        :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
    """

    def wrapper_function(
        *args: Any, pydantic_model: Optional[Type[T]] = None, **kwargs: Any
    ) -> Union[requests.Response, T]:
        try:
            r = f(*args, **kwargs)
            r.raise_for_status()
            if r.content:
                r_json = r.json()
                # Check if the API limit is exceeded
                if isinstance(r_json, dict):
                    message = r_json.get("message", "")
                    if message.startswith("You have exceeded"):
                        raise APILimitError(message)
                if pydantic_model is not None:
                    p = parse_obj_as(pydantic_model, r_json)
                    return p
            return r
        except requests.exceptions.Timeout as e:
            # Maybe set up for a retry, or continue in a retry loop
            raise TimeoutError(message=str(e))
        except requests.exceptions.HTTPError as e:
            raise NetworkError(message=str(e))
        except requests.exceptions.TooManyRedirects as e:
            # Tell the user their URL was bad and try a different one
            raise NetworkError(message="Bad url, please try a different one.", cause=e)
        except requests.exceptions.RequestException as e:
            logger.error(f"Unknown error: {e}")
            raise NetworkError(message=str(e))
        except pydantic.ValidationError as e:
            raise ValidationError(message=str(e))

    return wrapper_function


##########
# Client #
##########


class SegmentsClient:
    """A client with a connection to the Segments.ai platform.

    Note:
        Please refer to the `Python SDK quickstart <https://docs.segments.ai/tutorials/python-sdk-quickstart>`__ for a full example of working with the Python SDK.

    First install the SDK.

    >>> $ pip install --upgrade segments-ai

    Import the ``segments`` package in your python file and set up a client with an API key. An API key can be created on your `user account page <https://segments.ai/account>`__.

    >>> from segments import SegmentsClient
    >>> api_key = 'YOUR_API_KEY'
    >>> client = SegmentsClient(api_key)
    'Initialized successfully.'

    Or store your Segments API key in your environment (``SEGMENTS_API_KEY = 'YOUR_API_KEY'``):

    >>> from segments import SegmentsClient
    >>> client = SegmentsClient()
    'Found a Segments API key in your environment.'
    'Initialized successfully.'

    You can also use the client as a context manager:

    >>> with SegmentsClient() as client:
    >>>     client.get_datasets()


    Args:
        api_key: Your Segments.ai API key. If no API key given, reads ``SEGMENTS_API_KEY`` from the environment. Defaults to :obj:`None`.
        api_url: URL of the Segments.ai API. Defaults to ``https://api.segments.ai/``.
    Raises:
        :exc:`~segments.exceptions.AuthenticationError`: If an invalid API key is used or (when not passing the API key directly) if ``SEGMENTS_API_KEY`` is not found in your environment.
    """

    def __init__(
        self, api_key: Optional[str] = None, api_url: str = "https://api.segments.ai/"
    ):
        if api_key is None:
            self.api_key = os.getenv("SEGMENTS_API_KEY")
            if self.api_key is None:
                raise AuthenticationError(
                    message="Did you set SEGMENTS_API_KEY in your environment?"
                )
            else:
                logger.info("Found a Segments API key in your environment.")
        else:
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
            r = self._get("/api_status/?lib_version=0.58")  # TODO
            if r.status_code == 200:
                logger.info("Initialized successfully.")
        except NetworkError as e:
            if e.cause.response.status_code == 426:
                logger.info("The response HTTP status code is 426 Upgrade Required.")
                pass
            else:
                raise AuthenticationError(
                    message="Something went wrong. Did you use the right API key?"
                )

    # https://stackoverflow.com/questions/48160728/resourcewarning-unclosed-socket-in-python-3-unit-test
    def _close(self) -> None:
        """Close :class:`SegmentsClient` connections.

        You can manually close the Segments client's connections:

        >>> client = SegmentsClient()
        >>> client.get_datasets()
        >>> client.close()

        Or use the client as a context manager:

        >>> with SegmentsClient() as client:
        >>>     client.get_datasets()

        """
        self.api_session.close()
        self.s3_session.close()
        logger.info("Closed successfully.")

    # Use SegmentsClient as a context manager (e.g., with SegmentsClient() as client: client.add_dataset()).
    def __enter__(self) -> SegmentsClient:
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self._close()

    ############
    # Datasets #
    ############
    def get_datasets(self, user: Optional[str] = None) -> List[Dataset]:
        """Get a list of datasets.

        >>> datasets = client.get_datasets()
        >>> for dataset in datasets:
        >>>     print(dataset.name, dataset.description)

        Args:
            user: The user for which to get the datasets. Leave empty to get datasets of current user. Defaults to :obj:`None`.
        Returns:
            A list of datasets.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the datasets fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if user is not None:
            r = self._get(f"/users/{user}/datasets/", pydantic_model=List[Dataset])
        else:
            r = self._get("/user/datasets/", pydantic_model=List[Dataset])

        return r

    def get_dataset(self, dataset_identifier: str) -> Dataset:
        """Get a dataset.

        >>> dataset_identifier = 'jane/flowers'
        >>> dataset = client.get_dataset(dataset_identifier)
        >>> print(dataset)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
        Returns:
            A dataset.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the dataset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/datasets/{dataset_identifier}/", pydantic_model=Dataset)

        return r

    def add_dataset(
        self,
        name: str,
        description: Optional[str] = None,
        task_type: TaskType = "segmentation-bitmap",
        task_attributes: Optional[Dict[str, Any]] = None,
        category: Category = "other",
        public: bool = False,
        readme: Optional[str] = None,
        enable_skip_labeling: bool = True,
        enable_skip_reviewing: bool = False,
        enable_ratings: bool = False,
    ) -> Dataset:
        """Add a dataset.

        >>> dataset_name = 'flowers'
        >>> description = 'A dataset containing flowers of all kinds.'
        >>> task_type = 'segmentation-bitmap'
        >>> dataset = client.add_dataset(dataset_name, description, task_type)
        >>> print(dataset)

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
            description: The dataset description. Defaults to :obj:`None`.
            task_type: The dataset's task type. Defaults to ``segmentation-bitmap``.
            task_attributes: The dataset's task attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format>`__. Defaults to :obj:`None`.
            category: The dataset category. Defaults to ``other``.
            public: The dataset visibility. Defaults to :obj:`False`.
            readme: The dataset readme. Defaults to :obj:`None`.
            enable_skip_labeling: Enable the skip button in the labeling workflow. Defaults to :obj:`True`.
            enable_skip_reviewing: Enable the skip button in the reviewing workflow. Defaults to :obj:`False`.
            enable_ratings: Enable star-ratings for labeled images. Defaults to :obj:`False`.
        Returns:
            A newly created dataset.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the task attributes fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of the dataset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if description is None:
            description = ""

        if readme is None:
            readme = ""

        if task_attributes is None:
            task_attributes = {
                "format_version": "0.1",
                "categories": [{"id": 0, "name": "object"}],
            }

        try:
            TaskAttributes.parse_obj(task_attributes)
        except pydantic.ValidationError as e:
            logger.error(
                "Did you use the right task attributes? Please refer to the online documentation: https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format.",
            )
            raise ValidationError(message=str(e))
        else:
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
                "data_type": "IMAGE",
            }
            r = self._post("/user/datasets/", data=payload, pydantic_model=Dataset)

            return r

    def update_dataset(
        self,
        dataset_identifier: str,
        description: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        task_attributes: Optional[Dict[str, Any]] = None,
        category: Optional[Category] = None,
        public: Optional[bool] = None,
        readme: Optional[str] = None,
        enable_skip_labeling: Optional[bool] = None,
        enable_skip_reviewing: Optional[bool] = None,
        enable_ratings: Optional[bool] = None,
    ) -> Dataset:
        """Update a dataset.

        >>> dataset_identifier = 'jane/flowers'
        >>> description = 'A dataset containing flowers of all kinds.'
        >>> dataset = client.update_dataset(dataset_identifier, description)
        >>> print(dataset)

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
        Returns:
            An updated dataset.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the dataset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        payload: Dict[str, Any] = {}

        if description is not None:
            payload["description"] = description

        if task_type is not None:
            payload["task_type"] = task_type

        if task_attributes is not None:
            payload["task_attributes"] = task_attributes

        if category is not None:
            payload["category"] = category

        if public is not None:
            payload["public"] = public

        if readme is not None:
            payload["readme"] = readme

        if enable_skip_labeling is not None:
            payload["enable_skip_labeling"] = enable_skip_labeling

        if enable_skip_reviewing is not None:
            payload["enable_skip_reviewing"] = enable_skip_reviewing

        if enable_ratings is not None:
            payload["enable_ratings"] = enable_ratings

        r = self._patch(
            f"/datasets/{dataset_identifier}/", data=payload, pydantic_model=Dataset
        )
        logger.info(f"Updated {dataset_identifier}")

        return r

    def delete_dataset(self, dataset_identifier: str) -> None:
        """Delete a dataset.

        >>> dataset_identifier = 'jane/flowers'
        >>> client.delete_dataset(dataset_identifier)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.

        """

        self._delete(f"/datasets/{dataset_identifier}/")

    def add_dataset_collaborator(
        self, dataset_identifier: str, username: str, role: Role = "labeler"
    ) -> Collaborator:
        """Add a collaborator to a dataset.

        >>> dataset_identifier = 'jane/flowers'
        >>> username = 'john'
        >>> role = 'reviewer'
        >>> client.add_dataset_collaborator(dataset_identifier, username, role)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            username: The username of the collaborator to be added.
            role: The role of the collaborator to be added. Defaults to ``labeler``.
        Returns:
            A class containing the newly added collaborator with its role.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the collaborator fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        payload = {"user": username, "role": role}
        r = self._post(
            f"/datasets/{dataset_identifier}/collaborators/",
            data=payload,
            pydantic_model=Collaborator,
        )

        return r

    def delete_dataset_collaborator(
        self, dataset_identifier: str, username: str
    ) -> None:
        """Delete a dataset collaborator.

        >>> dataset_identifier = 'jane/flowers'
        >>> username = 'john'
        >>> client.delete_dataset_collaborator(dataset_identifier, username)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            username: The username of the collaborator to be deleted.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
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

            >>> dataset_identifier = 'jane/flowers'
            >>> samples = client.get_samples(dataset_identifier)
            >>> for sample in samples:
            >>>     print(sample.name, sample.uuid)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: Name to filter by. Defaults to :obj:`None`.(no filtering).
            label_status: Sequence of label statuses to filter by. Defaults to :obj:`None`.(no filtering).
            metadata: Sequence of 'key:value' metadata attributes to filter by. Defaults to :obj:`None`.(no filtering).
            sort: What to sort results by. One of ``name``, ``created``, ``priority``. Defaults to ``name``.
            direction: Sorting direction. One of ``asc`` (ascending) or ``desc`` (descending). Defaults to ``asc``.
            per_page: Pagination parameter indicating the maximum number of samples to return. Defaults to ``1000``.
            page: Pagination parameter indicating the page to return. Defaults to ``1``.
        Returns:
            A list of samples.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        # pagination
        query_string = f"?per_page={per_page}&page={page}"

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
            raise ValidationError(message=str(e))

        return results

    def get_sample(self, uuid: str, labelset: Optional[str] = None) -> Sample:
        """Get a sample.

        >>> uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
        >>> sample = client.get_sample(uuid)
        >>> print(sample)

        Args:
            uuid: The sample uuid.
            labelset: If defined, this additionally returns the label for the given labelset. Defaults to :obj:`None`.
        Returns:
            A sample
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        query_string = f"/samples/{uuid}/"

        if labelset is not None:
            query_string += f"?labelset={labelset}"

        r = self._get(query_string, pydantic_model=Sample)

        return r

    def add_sample(
        self,
        dataset_identifier: str,
        name: str,
        attributes: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0,
        embedding: Optional[Union[npt.NDArray[Any], List[float]]] = None,
    ) -> Sample:
        """Add a sample to a dataset.

        Note:
            - The content of the ``attributes`` field depends on the `sample type <https://docs.segments.ai/reference/sample-and-label-types/sample-types>`__.
            - If the image is on your local computer, you should first upload it to a cloud storage service like Amazon S3, Google Cloud Storage, Imgur, or `our asset storage service <https://docs.segments.ai/reference/python-sdk#upload-a-file-as-an-asset>`__.
            - If you create a sample with a URL from a public S3 bucket and you see an error on the platform, make sure to `properly configure your bucket's CORS settings <https://docs.aws.amazon.com/AmazonS3/latest/dev/cors.html>`__.

        >>> dataset_identifier = 'jane/flowers'
        >>> name = 'violet.jpg'
        >>> attributes = {
        ...     'image': {
        ...         'url': 'https://example.com/violet.jpg'
        ...     }
        ... }
        >>> # Metadata and priority are optional fields.
        >>> metadata = {
        ...     'city': 'London',
        ...     'weather': 'cloudy',
        ...     'robot_id': 3
        ... }
        >>> priority = 10 # Samples with higher priority value will be labeled first. Default is 0.
        >>> sample = client.add_sample(dataset_identifier, name, attributes, metadata, priority)
        >>> print(sample)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the sample.
            attributes: The sample attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/sample-types>`__.
            metadata: Any sample metadata. Example: ``{'weather': 'sunny', 'camera_id': 3}``.
            priority: Priority in the labeling queue. Samples with higher values will be labeled first. Defaults to ``0``.
            embedding: Embedding of this sample represented by an array of floats.
        Returns:
            A newly created sample.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the sample attributes fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        try:
            parse_obj_as(SampleAttributes, attributes)
        except pydantic.ValidationError as e:
            logger.error(
                "Did you use the right sample attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/sample-types.",
            )
            raise ValidationError(message=str(e))
        else:
            payload: Dict[str, Any] = {
                "name": name,
                "attributes": attributes,
            }

            if metadata is not None:
                payload["metadata"] = metadata

            if priority is not None:
                payload["priority"] = priority

            if embedding is not None:
                payload["embedding"] = embedding

            r = self._post(
                f"/datasets/{dataset_identifier}/samples/",
                data=payload,
                pydantic_model=Sample,
            )
            logger.info(f"Added {name}")

            return r

    def update_sample(
        self,
        uuid: str,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0,
        embedding: Optional[Union[npt.NDArray[Any], List[float]]] = None,
    ) -> Sample:
        """Update a sample.

        >>> uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
        >>> metadata = {
        ...     'city': 'London',
        ...     'weather': 'cloudy',
        ...     'robot_id': 3
        ... }
        >>> priority = 10 # Samples with higher priority value will be labeled first. Default is 0.
        >>> sample = client.update_sample(uuid, metadata=metadata, priority=priority)
        >>> print(sample)

        Args:
            uuid: The sample uuid.
            name: The name of the sample.
            attributes: The sample attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/sample-types>`__.
            metadata: Any sample metadata. Example: ``{'weather': 'sunny', 'camera_id': 3}``.
            priority: Priority in the labeling queue. Samples with higher values will be labeled first. Default is ``0``.
            embedding: Embedding of this sample represented by list of floats.
        Returns:
            An updated sample.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.ValidationError`: If validation of the samples fails.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        payload: Dict[str, Any] = {}

        if name is not None:
            payload["name"] = name

        if attributes is not None:
            payload["attributes"] = attributes

        if metadata is not None:
            payload["metadata"] = metadata

        if priority is not None:
            payload["priority"] = priority

        if embedding is not None:
            payload["embedding"] = embedding

        r = self._patch(f"/samples/{uuid}/", data=payload, pydantic_model=Sample)
        logger.info(f"Updated {uuid}")

        return r

    def delete_sample(self, uuid: str) -> None:
        """Delete a sample.

        >>> uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
        >>> client.delete_sample(uuid)

        Args:
            uuid: The sample uuid.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/samples/{uuid}/")

    ##########
    # Labels #
    ##########
    def get_label(self, sample_uuid: str, labelset: str = "ground-truth") -> Label:
        """Get a label.

        >>> sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
        >>> label = client.get_label(sample_uuid)
        >>> print(label)

        Args:
            sample_uuid: The sample uuid.
            labelset: The labelset this label belongs to. Defaults to ``ground-truth```.
        Returns:
            A label.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(f"/labels/{sample_uuid}/{labelset}/", pydantic_model=Label)

        return r

    def add_label(
        self,
        sample_uuid: str,
        attributes: Dict[str, Any],
        labelset: str = "ground-truth",
        label_status: LabelStatus = "PRELABELED",
        score: Optional[float] = None,
    ) -> Label:
        """Add a label to a sample.

        A label is added to a sample in relation to a `label set`, such as the default `ground-truth` label set, or a newly created label set for `uploading model predictions <https://docs.segments.ai/guides/upload-model-predictions>`__. You can create a new label set by clicking the "Add new label set" link on the Samples tab.

        Note:
            The content of the ``attributes`` field depends on the `label type <https://docs.segments.ai/reference/sample-and-label-types/label-types>`__.

        >>> sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
        >>> attributes = {
        ...     'format_version': '0.1',
        ...     'annotations': [
        ...         {
        ...             'id': 1,
        ...             'category_id': 1,
        ...             'type': 'bbox',
        ...             'points': [
        ...                 [12.34, 56.78],
        ...                 [90.12, 34.56]
        ...             ]
        ...         },
        ...     ]
        ... }
        >>> client.add_label(sample_uuid, attributes)

        Args:
            sample_uuid: The sample uuid.
            attributes: The label attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/label-types>`__.
            labelset: The labelset this label belongs to. Defaults to ``ground-truth``.
            label_status: The label status. Defaults to ``PRELABELED``.
            score: The label score. Defaults to :obj:`None`.
        Returns:
            A newly created label.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the attributes fails.
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        try:
            parse_obj_as(LabelAttributes, attributes)
        except pydantic.ValidationError as e:
            logger.error(
                "Did you use the right label attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/label-types.",
            )
            raise ValidationError(message=str(e))
        else:
            payload: Dict[str, Any] = {
                "label_status": label_status,
                "attributes": attributes,
            }

            if score is not None:
                payload["score"] = score

            r = self._put(
                f"/labels/{sample_uuid}/{labelset}/", data=payload, pydantic_model=Label
            )

            return r

    def update_label(
        self,
        sample_uuid: str,
        attributes: Dict[str, Any],
        labelset: str = "ground-truth",
        label_status: LabelStatus = "PRELABELED",
        score: Optional[float] = None,
    ) -> Label:
        """Update a label.

        >>> sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
        >>> attributes = {
        ...     'format_version': '0.1',
        ...     'annotations': [
        ...         {
        ...             'id': 1,
        ...             'category_id': 1,
        ...             'type': 'bbox',
        ...             'points': [
        ...                 [12.34, 56.78],
        ...                 [90.12, 34.56]
        ...             ]
        ...         },
        ...     ]
        ... }
        >>> client.update_label(sample_uuid, attributes)

        Args:
            sample_uuid: The sample uuid.
            attributes: The label attributes. Please refer to the `online documentation <https://docs.segments.ai/reference/sample-and-label-types/label-types>`__.
            labelset: The labelset this label belongs to. Defaults to ``ground-truth``.
            label_status: The label status. Defaults to ``PRELABELED``.
            score: The label score. Defaults to :obj:`None`.
        Returns:
            An updated label.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the label fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        payload: Dict[str, Any] = {}

        if attributes is not None:
            payload["attributes"] = attributes

        if label_status is not None:
            payload["label_status"] = label_status

        if score is not None:
            payload["score"] = score

        r = self._patch(
            f"/labels/{sample_uuid}/{labelset}/", data=payload, pydantic_model=Label
        )

        return r

    def delete_label(self, sample_uuid: str, labelset: str = "ground-truth") -> None:
        """Delete a label.

        >>> sample_uuid = '602a3eec-a61c-4a77-9fcc-3037ce5e9606'
        >>> labelset = 'ground-truth'
        >>> client.delete_label(sample_uuid, labelset)

        Args:
            sample_uuid: The sample uuid.
            labelset: The labelset this label belongs to. Defaults to ``ground-truth``.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/labels/{sample_uuid}/{labelset}/")

    #############
    # Labelsets #
    #############
    def get_labelsets(self, dataset_identifier: str) -> List[Labelset]:
        """Get the labelsets in a dataset.

        >>> dataset_identifier = 'jane/flowers'
        >>> labelsets = client.get_labelsets(dataset_identifier)
        >>> for labelset in labelsets:
        >>>     print(labelset.name, labelset.description)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
        Returns:
            A list of labelsets.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the labelsets fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(
            f"/datasets/{dataset_identifier}/labelsets/", pydantic_model=List[Labelset]
        )

        return r

    def get_labelset(self, dataset_identifier: str, name: str) -> Labelset:
        """Get a labelset.

        >>> dataset_identifier = 'jane/flowers'
        >>> name = 'model-predictions'
        >>> labelset = client.get_labelset(dataset_identifier, name)
        >>> print(labelset)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the labelset.
        Returns:
            A labelset.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the labelset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(
            f"/datasets/{dataset_identifier}/labelsets/{name}/", pydantic_model=Labelset
        )

        return r

    def add_labelset(
        self, dataset_identifier: str, name: str, description: Optional[str] = None
    ) -> Labelset:
        """Add a labelset to a dataset.

        >>> dataset_identifier = 'jane/flowers'
        >>> name = 'model-predictions-resnet50'
        >>> client.add_labelset(dataset_identifier, name)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the labelset.
            description: The labelset description. Defaults to :obj:`None`.
        Returns:
            A labelset.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the labelset fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if description is None:
            description = ""

        payload = {
            "name": name,
            "description": description,
            "attributes": "{}",
        }
        r = self._post(
            f"/datasets/{dataset_identifier}/labelsets/",
            data=payload,
            pydantic_model=Labelset,
        )

        return r

    def delete_labelset(self, dataset_identifier: str, name: str) -> None:
        """Delete a labelset.

        >>> dataset_identifier = 'jane/flowers'
        >>> name = 'model-predictions'
        >>> client.delete_labelset(dataset_identifier, name)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the labelset.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        self._delete(f"/datasets/{dataset_identifier}/labelsets/{name}/")

    ############
    # Releases #
    ############
    def get_releases(self, dataset_identifier: str) -> List[Release]:
        """Get the releases in a dataset.

        >>> dataset_identifier = 'jane/flowers'
        >>> releases = client.get_releases(dataset_identifier)
        >>> for release in releases:
        >>>     print(release.name, release.description, release.attributes.url)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
        Returns:
            A list of releases.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the releases fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(
            f"/datasets/{dataset_identifier}/releases/", pydantic_model=List[Release]
        )

        return r

    def get_release(self, dataset_identifier: str, name: str) -> Release:
        """Get a release.

        >>> dataset_identifier = 'jane/flowers'
        >>> name = 'v0.1'
        >>> release = client.get_release(dataset_identifier, name)
        >>> print(release)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the release.
        Returns:
            A release.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the release fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        r = self._get(
            f"/datasets/{dataset_identifier}/releases/{name}/", pydantic_model=Release
        )

        return r

    def add_release(
        self, dataset_identifier: str, name: str, description: Optional[str] = None
    ) -> Release:
        """Add a release to a dataset.

        >>> dataset_identifier = 'jane/flowers'
        >>> name = 'v0.1'
        >>> description = 'My first release.'
        >>> release = client.add_release(dataset_identifier, name, description)
        >>> print(release)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the release.
            description: The release description. Defaults to :obj:`None`.
        Returns:
            A newly created release.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the release fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """

        if description is None:
            description = ""

        payload = {"name": name, "description": description}
        r = self._post(
            f"/datasets/{dataset_identifier}/releases/",
            data=payload,
            pydantic_model=Release,
        )

        return r

    def delete_release(self, dataset_identifier: str, name: str) -> None:
        """Delete a release.

        >>> dataset_identifier = 'jane/flowers'
        >>> name = 'v0.1'
        >>> client.delete_release(dataset_identifier, name)

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: ``jane/flowers``.
            name: The name of the release.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
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

        >>> filename = '/home/jane/flowers/violet.jpg'
        >>> with open(filename, 'rb') as f:
        >>>     filename = 'violet.jpg'
        >>>     asset = client.upload_asset(f, filename)
        >>> image_url = asset.url
        >>> print(image_url)

        Args:
            file: A file object.
            filename: The file name. Defaults to ``label.png``.
        Returns:
            A class representing the uploaded file.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the file fails.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
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
            raise ValidationError(message=str(e))

        return f

    ####################
    # Helper functions #
    ####################
    @exception_handler
    def _get(
        self,
        endpoint: str,
        auth: bool = True,
        pydantic_model: Optional[T] = None,
    ) -> requests.Response:
        """Send a GET request.

        Args:
            endpoint: The API endpoint.
            auth: If we want to authorize the request. Defaults to :obj:`True`.
            pydantic_model: The pydantic class to parse the JSON response into. Defaults to :obj:`None`.
        Returns:
            The ``requests`` library response.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error) - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """

        headers = self._get_auth_header() if auth else None

        r = self.api_session.get(
            urllib.parse.urljoin(self.api_url, endpoint), headers=headers
        )

        return r

    @exception_handler
    def _post(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        pydantic_model: Optional[T] = None,
    ) -> requests.Response:
        """Send a POST request.

        Args:
            endpoint: The API endpoint.
            data: The JSON data. Defaults to :obj:`None`.
            auth: If we want to authorize the request with the API key. Defaults to :obj:`True`.
            pydantic_model: The pydantic class to parse the JSON response into. Defaults to :obj:`None`.
        Returns:
            The ``requests`` library response.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error) - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """
        headers = self._get_auth_header() if auth else None

        r = self.api_session.post(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )

        return r

    @exception_handler
    def _put(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        pydantic_model: Optional[T] = None,
    ) -> requests.Response:
        """Send a PUT request.

        Args:
            endpoint: The API endpoint.
            data: The JSON data. Defaults to :obj:`None`.
            auth: If we want to authorize the request with the API key. Defaults to :obj:`True`.
            pydantic_model: The pydantic class to parse the JSON response into. Defaults to :obj:`None`.
        Returns:
            The ``requests`` library response.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error) - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """
        headers = self._get_auth_header() if auth else None

        r = self.api_session.put(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )

        return r

    @exception_handler
    def _patch(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        pydantic_model: Optional[T] = None,
    ) -> requests.Response:
        """Send a PATCH request.

        Args:
            endpoint: The API endpoint.
            data: The JSON data. Defaults to :obj:`None`.
            auth: If we want to authorize the request with the API key. Defaults to :obj:`True`.
            pydantic_model: The pydantic class to parse the JSON response into. Defaults to :obj:`None`.
        Returns:
            The ``requests`` library response.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error) - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
            :exc:`~segments.exceptions.TimeoutError`: If the request times out - catches :exc:`requests.exceptions.TimeoutError`.
        """
        headers = self._get_auth_header() if auth else None

        r = self.api_session.patch(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )

        return r

    @exception_handler
    def _delete(
        self,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        pydantic_model: Optional[T] = None,
    ) -> requests.Response:
        """Send a DELETE request.

        Args:
            endpoint: The API endpoint.
            data: The JSON data. Defaults to :obj:`None`.
            auth: If we want to authorize the request with the API key. Defaults to :obj:`True`.
            pydantic_model: The pydantic class to parse the JSON response into. Defaults to :obj:`None`.
        Returns:
            The ``requests`` library response.
        Raises:
            :exc:`~segments.exceptions.ValidationError`: If validation of the response fails - catches :exc:`pydantic.ValidationError`.
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error) - catches :exc:`requests.HTTPError` and catches :exc:`requests.RequestException`.
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
        """Get the authorization header with the API key.

        Returns:
            The authorization header or :obj:`None`.if there is no API key.
        """
        return {"Authorization": f"APIKey {self.api_key}"} if self.api_key else None

    @exception_handler
    def _upload_to_aws(
        self, file: Union[TextIO, BinaryIO], url: str, aws_fields: AWSFields
    ) -> requests.Response:
        """Upload file to AWS.

        Args:
            file: The file we want to upload.
            url: The request's url.
            aws_fields: The AWS fields.
        Returns:
            The ``requests`` library response.
        Raises:
            :exc:`~segments.exceptions.APILimitError`: If the API limit is exceeded.
            :exc:`~segments.exceptions.NetworkError`: If the response status code is 4XX (client error) or 5XX (server error).
            :exc:`~segments.exceptions.TimeoutError`: If the request times out.
        """
        files = {"file": file}
        r = self.s3_session.post(url, files=files, data=aws_fields)

        return r