from typing import IO, Any, Dict, List, Optional, Union
from typing_extensions import Literal
import urllib.parse
import requests
from pydantic import parse_obj_as
from .typehints import (
    AWSFields,
    AuthHeader,
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
    Category,
)

# import numpy.typing as npt


class SegmentsClient:
    """SegmentsClient class.

    Args:
        api_key: Your Segments.ai API key.
        api_url: URL of the Segments.ai API.

    Attributes:
        api_key: Your Segments.ai API key.
        api_url: URL of the Segments.ai API.

    """

    def __init__(self, api_key: str, api_url: str = "https://api.segments.ai/"):
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

        r = self.get("/api_status/?lib_version=0.58")
        if r.status_code == 200:
            print("Initialized successfully.")
        elif r.status_code == 426:
            pass
        else:
            raise Exception("Something went wrong. Did you use the right api key?")

    # https://stackoverflow.com/questions/48160728/resourcewarning-unclosed-socket-in-python-3-unit-test
    def close(self) -> None:
        self.api_session.close()
        self.s3_session.close()
        print("Closed successfully.")

    ############
    # Datasets #
    ############
    def get_datasets(self, user: Optional[str] = None) -> List[Dataset]:
        """Get a list of datasets.

        Args:
            user: The user for which to get the datasets. Leave empty to get datasets of current user. Defaults to None.

        Returns:
            A list of classes representing the datasets.
        """

        if user is not None:
            r = self.get("/users/{}/datasets/".format(user))
        else:
            r = self.get("/user/datasets/")
        # datasets = [
        #     from_dict(data_class=Dataset, data=dataset, config=DACITE_CONFIG)
        #     for dataset in r.json()
        # ]
        datasets = parse_obj_as(List[Dataset], r.json())

        return datasets

    def get_dataset(self, dataset_identifier: str) -> Dataset:
        """Get a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            Dataset: a class representing the dataset.
        """

        r = self.get("/datasets/{}/".format(dataset_identifier))
        # dataset = from_dict(data_class=Dataset, data=r.json(), config=DACITE_CONFIG)
        dataset = Dataset.parse_obj(r.json())

        return dataset

    def add_dataset(
        self,
        name: str,
        description: str = "",
        task_type: TaskType = "segmentation-bitmap",
        task_attributes: Optional[Dict[str, Any]] = None,
        category: Category = "other",
        public: bool = False,
        readme: str = "",
        enable_skip_labeling: bool = True,
        enable_skip_reviewing: bool = False,
        enable_ratings: bool = False,
    ) -> Dataset:
        """Add a dataset.

        Args:
            name: The dataset name. Example: flowers.
            description: The dataset description. Defaults to ''.
            task_type: The dataset's task type. One of 'segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes', 'keypoints'. Defaults to 'segmentation-bitmap', 'pointcloud-segmentation', 'pointcloud-detection'.
            task_attributes: The dataset's task attributes. Defaults to None.
            category: The dataset category. Defaults to 'other'.
            public: The dataset visibility. Defaults to False.
            readme: The dataset readme. Defaults to ''.
            enable_skip_labeling: Enable the skip button in the labeling workflow. Defaults to True.
            enable_skip_reviewing: Enable the skip button in the reviewing workflow. Defaults to False.
            enable_ratings: Enable star-ratings for labeled images. Defaults to False.

        Returns:
            A class representing the newly created dataset.
        """

        if task_attributes is None:
            task_attributes = {
                "format_version": "0.1",
                "categories": [{"id": 0, "name": "object"}],
            }
        try:
            # from_dict(
            #     data_class=TaskAttributes, data=task_attributes, config=DACITE_CONFIG
            # )
            TaskAttributes.parse_obj(task_attributes)
        except Exception as e:
            raise Exception(
                e,
                "Did you use the right task attributes? Please refer to the online documentation: https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format.",
            )
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
            r = self.post("/user/datasets/", payload)
            # dataset = from_dict(data_class=Dataset, data=r.json(), config=DACITE_CONFIG)
            dataset = Dataset.parse_obj(r.json())

            return dataset

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

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            description: The dataset description.
            task_type: The dataset's task type. One of 'segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes', 'keypoints', 'pointcloud-segmentation', 'pointcloud-detection'.
            task_attributes: The dataset's task attributes.
            category: The dataset category.
            public: The dataset visibility.
            readme: The dataset readme.
            enable_skip_labeling: Enable the skip button in the labeling workflow.
            enable_skip_reviewing: Enable the skip button in the reviewing workflow.
            enable_ratings: Enable star-ratings for labeled images.

        Returns:
            A class representing the updated dataset.
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

        r = self.patch("/datasets/{}/".format(dataset_identifier), payload)
        print("Updated " + dataset_identifier)
        # dataset = from_dict(data_class=Dataset, data=r.json(), config=DACITE_CONFIG)
        dataset = Dataset.parse_obj(r.json())

        return dataset

    def delete_dataset(self, dataset_identifier: str) -> None:
        """Delete a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
        """

        r = self.delete("/datasets/{}/".format(dataset_identifier))

    def add_dataset_collaborator(
        self, dataset_identifier: str, username: str, role: Role = "labeler"
    ) -> Collaborator:
        """Add a collaborator to a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            username: The username of the collaborator to be added.
            role: The role of the collaborator to be added. One of labeler, reviewer, admin. Defaults to labeler.

        Returns:
            A class containing the newly added collaborator with its role.
        """
        payload = {"user": username, "role": role}
        r = self.post("/datasets/{}/collaborators/".format(dataset_identifier), payload)
        # collaborator = from_dict(
        #     data_class=Collaborator, data=r.json(), config=DACITE_CONFIG
        # )
        collaborator = Collaborator.parse_obj(r.json())

        return collaborator

    def delete_dataset_collaborator(
        self, dataset_identifier: str, username: str
    ) -> None:
        """Delete a dataset collaborator.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            username: The username of the collaborator to be deleted.
        """
        r = self.delete(
            "/datasets/{}/collaborators/{}".format(dataset_identifier, username),
        )

    ###########
    # Samples #
    ###########
    def get_samples(
        self,
        dataset_identifier: str,
        name: Optional[str] = None,
        label_status: Optional[Union[LabelStatus, List[LabelStatus]]] = None,
        metadata: Optional[Union[List[str], str]] = None,
        sort: Literal["name", "created", "priority"] = "name",
        direction: Literal["asc", "desc"] = "asc",
        per_page: int = 1000,
        page: int = 1,
    ) -> List[Sample]:
        """Get the samples in a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name: Name to filter by. Defaults to None (no filtering).
            label_status: List of label statuses to filter by. Defaults to None (no filtering).
            metadata: List of 'key:value' metadata attributes to filter by. Defaults to None (no filtering).
            sort: What to sort results by. One of 'name', 'created', 'priority'. Defaults to 'name'.
            direction: Sorting direction. One of 'asc' (ascending) or 'desc' (descending). Defaults to 'asc'.
            per_page: Pagination parameter indicating the maximum number of samples to return. Defaults to 1000.
            page: Pagination parameter indicating the page to return. Defaults to 1.

        Returns:
            A list of classes representing the samples.
        """

        # pagination
        query_string = "?per_page={}&page={}".format(per_page, page)

        # filter by name
        if name is not None:
            query_string += "&name__contains={}".format(name)

        # filter by metadata
        if metadata is not None:
            if isinstance(metadata, str):
                metadata = [metadata]
            query_string += "&filters={}".format(",".join(metadata))

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
            query_string += "&sort={}{}".format(direction_str, sort_str)

        r = self.get("/datasets/{}/samples/{}".format(dataset_identifier, query_string))
        results = r.json()

        for result in results:
            result.pop("label", None)

        results = parse_obj_as(List[Sample], results)

        return results

    def get_sample(self, uuid: str, labelset: Optional[str] = None) -> Sample:
        """Get a sample.

        Args:
            uuid: The sample uuid.
            labelset: If defined, this additionally returns the label for the given labelset. Defaults to None.

        Returns:
            A class representing the sample
        """

        query_string = "/samples/{}/".format(uuid)

        if labelset is not None:
            query_string += "?labelset={}".format(labelset)

        r = self.get(query_string)
        sample = Sample.parse_obj(r.json())

        return sample

    def add_sample(
        self,
        dataset_identifier: str,
        name: str,
        attributes: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0,
        embedding: Any = None,  # embedding: Optional[Union[npt.NDArray[Any], List[float]]] = None
    ) -> Sample:
        """Add a sample to a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name: The name of the sample.
            attributes: The sample attributes. Please refer to the online documentation.
            metadata: Any sample metadata. Example: {'weather': 'sunny', 'camera_id': 3}.
            priority: Priority in the labeling queue. Samples with higher values will be labeled first. Defaults to 0.
            embedding: Embedding of this sample represented by an array of floats.

        Returns:
            A class representing the newly created sample.
        """

        try:
            parse_obj_as(SampleAttributes, attributes)
        except Exception as e:
            raise Exception(
                e,
                "Did you use the right sample attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/sample-types.",
            )
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

            r = self.post("/datasets/{}/samples/".format(dataset_identifier), payload)
            print("Added " + name)
            sample = Sample.parse_obj(r.json())

            return sample

    def update_sample(
        self,
        uuid: str,
        name: Optional[str] = None,
        attributes: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0,
        embedding: Optional[
            List[float]
        ] = None,  # Optional[Union[npt.NDArray[Any], List[float]]] = None,
    ) -> Sample:
        """Update a sample.

        Args:
            uuid: The sample uuid.
            name: The name of the sample.
            attributes: The sample attributes. Please refer to the online documentation.
            metadata: Any sample metadata. Example: {'weather': 'sunny', 'camera_id': 3}.
            priority: Priority in the labeling queue. Samples with higher values will be labeled first. Default is 0.
            embedding: Embedding of this sample represented by an array of floats.

        Returns:
            A class representing the updated sample.
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

        r = self.patch("/samples/{}/".format(uuid), payload)
        print("Updated " + uuid)
        # sample = from_dict(data_class=Sample, data=r.json(), config=DACITE_CONFIG)
        sample = Sample.parse_obj(r.json())

        return sample

    def delete_sample(self, uuid: str) -> None:
        """Delete a sample.

        Args:
            uuid: The sample uuid.
        """

        r = self.delete("/samples/{}/".format(uuid))

    ##########
    # Labels #
    ##########
    def get_label(self, sample_uuid: str, labelset: str = "ground-truth") -> Label:
        """Get a label.

        Args:
            sample_uuid: The sample uuid.
            labelset: The labelset this label belongs to. Defaults to 'ground-truth'.

        Returns:
            A class representing the label.
        """

        r = self.get("/labels/{}/{}/".format(sample_uuid, labelset))
        label = Label.parse_obj(r.json())

        return label

    def add_label(
        self,
        sample_uuid: str,
        labelset: str,
        attributes: Dict[str, Any],
        label_status: LabelStatus = "PRELABELED",
        score: Optional[float] = None,
    ) -> Label:
        """Add a label to a sample.

        Args:
            sample_uuid: The sample uuid.
            labelset: The labelset this label belongs to.
            attributes: The label attributes. Please refer to the online documentation.
            label_status: The label status. Defaults to 'PRELABELED'.
            score: The label score. Defaults to None.

        Returns:
            A class representing the newly created label.
        """

        try:
            parse_obj_as(LabelAttributes, attributes)
        except Exception as e:
            raise Exception(
                e,
                "Did you use the right label attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/label-types.",
            )
        else:
            payload: Dict[str, Any] = {
                "label_status": label_status,
                "attributes": attributes,
            }

            if score is not None:
                payload["score"] = score

            r = self.put("/labels/{}/{}/".format(sample_uuid, labelset), payload)
            label = Label.parse_obj(r.json())

            return label

    def update_label(
        self,
        sample_uuid: str,
        labelset: str,
        attributes: Dict[str, Any],
        label_status: LabelStatus = "PRELABELED",
        score: Optional[float] = None,
    ) -> Label:
        """Update a label.

        Args:
            sample_uuid: The sample uuid.
            labelset: The labelset this label belongs to.
            attributes: The label attributes. Please refer to the online documentation.
            label_status: The label status. Defaults to 'PRELABELED'.
            score: The label score. Defaults to None.

        Returns:
            A class representing the updated label.
        """

        payload: Dict[str, Any] = {}

        if attributes is not None:
            payload["attributes"] = attributes

        if label_status is not None:
            payload["label_status"] = label_status

        if score is not None:
            payload["score"] = score

        r = self.patch("/labels/{}/{}/".format(sample_uuid, labelset), payload)
        label = Label.parse_obj(r.json())

        return label

    def delete_label(self, sample_uuid: str, labelset: str) -> None:
        """Delete a label.

        Args:
            sample_uuid: The sample uuid.
            labelset: The labelset this label belongs to.
        """

        r = self.delete("/labels/{}/{}/".format(sample_uuid, labelset))

    #############
    # Labelsets #
    #############
    def get_labelsets(self, dataset_identifier: str) -> List[Labelset]:
        """Get the labelsets in a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            A list of classes representing the labelsets.
        """

        r = self.get("/datasets/{}/labelsets/".format(dataset_identifier))
        labelsets = parse_obj_as(List[Labelset], r.json())

        return labelsets

    def get_labelset(self, dataset_identifier: str, name: str) -> Labelset:
        """Get a labelset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name: The name of the labelset.

        Returns:
            A class representing the labelset.
        """

        r = self.get("/datasets/{}/labelsets/{}/".format(dataset_identifier, name))
        labelset = Labelset.parse_obj(r.json())

        return labelset

    def add_labelset(
        self, dataset_identifier: str, name: str, description: str = ""
    ) -> Labelset:
        """Add a labelset to a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name: The name of the labelset.
            description: The labelset description.

        Returns:
            A class representing the labelset.
        """

        payload = {
            "name": name,
            "description": description,
            "attributes": "{}",
        }
        r = self.post("/datasets/{}/labelsets/".format(dataset_identifier), payload)
        labelset = Labelset.parse_obj(r.json())

        return labelset

    def delete_labelset(self, dataset_identifier: str, name: str) -> None:
        """Delete a labelset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name: The name of the labelset.
        """

        r = self.delete("/datasets/{}/labelsets/{}/".format(dataset_identifier, name))

    ############
    # Releases #
    ############
    def get_releases(self, dataset_identifier: str) -> List[Release]:
        """Get the releases in a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            A list of classes representing the releases.
        """

        r = self.get("/datasets/{}/releases/".format(dataset_identifier))
        releases = parse_obj_as(List[Release], r.json())

        return releases

    def get_release(self, dataset_identifier: str, name: str) -> Release:
        """Get a release.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name: The name of the release.

        Returns:
            A class representing the release.
        """

        r = self.get("/datasets/{}/releases/{}/".format(dataset_identifier, name))
        release = Release.parse_obj(r.json())

        return release

    def add_release(
        self, dataset_identifier: str, name: str, description: str = ""
    ) -> Release:
        """Add a release to a dataset.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name: The name of the release.
            description: The release description.

        Returns:
            A class representing the newly created release.
        """

        payload = {"name": name, "description": description}
        r = self.post("/datasets/{}/releases/".format(dataset_identifier), payload)
        print(r.json())
        release = Release.parse_obj(r.json())

        return release

    def delete_release(self, dataset_identifier: str, name: str) -> None:
        """Delete a release.

        Args:
            dataset_identifier: The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name: The name of the release.
        """

        r = self.delete("/datasets/{}/releases/{}/".format(dataset_identifier, name))

    ##########
    # Assets #
    ##########
    def upload_asset(self, file: IO, filename: str = "label.png") -> File:
        """Upload an asset.

        Args:
            file: A file object.
            filename: The file name. Defaults to label.png.

        Returns:
            A class representing the uploaded file.
        """

        r = self.post("/assets/", {"filename": filename})
        presigned_post_fields = PresignedPostFields.parse_obj(
            r.json()["presignedPostFields"]
        )
        response_aws = self._upload_to_aws(
            file, presigned_post_fields.url, presigned_post_fields.fields
        )
        f = File.parse_obj(r.json())

        return f

    # Error handling: https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module
    def _get_auth_header(self) -> Optional[AuthHeader]:
        if self.api_key:
            return {"Authorization": "APIKey {}".format(self.api_key)}
        else:
            return None

    def get(self, endpoint: str, auth: bool = True) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        r = self.api_session.get(
            urllib.parse.urljoin(self.api_url, endpoint), headers=headers
        )
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{} | {}".format(errh, r.json()))

        return r

    def post(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, auth: bool = True
    ) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        r = self.api_session.post(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{} | {}".format(errh, r.json()))

        return r

    def put(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, auth: bool = True
    ) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        r = self.api_session.put(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{} | {}".format(errh, r.json()))

        return r

    def patch(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, auth: bool = True
    ) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        r = self.api_session.patch(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{} | {}".format(errh, r.json()))

        return r

    def delete(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, auth: bool = True
    ) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        r = self.api_session.delete(
            urllib.parse.urljoin(self.api_url, endpoint),
            json=data,  # data=data
            headers=headers,
        )
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{}".format(errh))

        return r

    def _upload_to_aws(
        self, file: IO, url: str, aws_fields: AWSFields
    ) -> requests.Response:
        files = {"file": file}
        r = self.s3_session.post(url, files=files, data=aws_fields)

        return r
