from dacite import from_dict
from typing import IO, Any, Dict, List, Optional, Union
from typing_extensions import Literal
import urllib.parse
import requests
import json
import nptyping as npt

from typehints import (
    DACITE_CONFIG,
    AWSFields,
    AuthHeader,
    Collaborator,
    Dataset,
    File,
    Label,
    LabelAttributes,
    LabelSet,
    LabelStatus,
    Release,
    Role,
    Sample,
    SampleAttributes,
    TaskAttributes,
    TaskType,
)


class SegmentsClient:
    """SegmentsClient class.

    Args:
        api_key (str): Your Segments.ai API key.
        api_url (str, optional): URL of the Segments.ai API.

    Attributes:
        api_key (str): Your Segments.ai API key.
        api_url (str): URL of the Segments.ai API.

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

    ############
    # Datasets #
    ############
    def get_datasets(self, user: Optional[str] = None) -> List[Dataset]:
        """Get a list of datasets.

        Args:
            user (str, optional): The user for which to get the datasets. Leave empty to get datasets of current user. Defaults to None.

        Returns:
            list: a list of dictionaries representing the datasets.
        """

        if user is not None:
            r = self.get("/users/{}/datasets/".format(user))
        else:
            r = self.get("/user/datasets/")

        # with open(
        #     "/Users/arnouthillen/segments-ai/segments2/tests/get_datasets.json", "w"
        # ) as f:
        #     print(json.dump(r.json(), f))

        datasets = [
            from_dict(data_class=Dataset, data=dataset, config=DACITE_CONFIG)
            for dataset in r.json()
        ]
        return datasets

    def get_dataset(self, dataset_identifier: str) -> Dataset:
        """Get a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            dict: a dictionary representing the dataset.
        """

        r = self.get("/datasets/{}/".format(dataset_identifier))

        # with open(
        #     "/Users/arnouthillen/segments-ai/segments2/tests/get_dataset.json", "w"
        # ) as f:
        #     print(json.dump(r.json(), f))

        dataset = from_dict(data_class=Dataset, data=r.json(), config=DACITE_CONFIG)
        return dataset

    def add_dataset(
        self,
        name: str,
        description: str = "",
        task_type: TaskType = "segmentation-bitmap",
        task_attributes: Optional[Dict[str, Any]] = None,
        category: str = "other",
        public: bool = False,
        readme: str = "",
        enable_skip_labeling: bool = True,
        enable_skip_reviewing: bool = False,
        enable_ratings: bool = False,
    ) -> Dataset:
        """Add a dataset.

        Args:
            name (str): The dataset name. Example: flowers.
            description (str, optional): The dataset description. Defaults to ''.
            task_type (str, optional): The dataset's task type. One of 'segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes', 'keypoints'. Defaults to 'segmentation-bitmap', 'pointcloud-segmentation', 'pointcloud-detection'.
            task_attributes (dict, optional): The dataset's task attributes. Defaults to None.
            category (str, optional): The dataset category. Defaults to 'other'.
            public (bool, optional): The dataset visibility. Defaults to False.
            readme (str, optional): The dataset readme. Defaults to ''.
            enable_skip_labeling (bool, optional): Enable the skip button in the labeling workflow. Defaults to True.
            enable_skip_reviewing (bool, optional): Enable the skip button in the reviewing workflow. Defaults to False.
            enable_ratings: Enable star-ratings for labeled images. Defaults to False.

        Returns:
            dict: a dictionary representing the newly created dataset.
        """

        if task_attributes is None:
            task_attributes = {
                "format_version": "0.1",
                "categories": [{"id": 0, "name": "object"}],
            }
        try:
            from_dict(
                data_class=TaskAttributes, data=task_attributes, config=DACITE_CONFIG
            )
        except Exception as e:
            print(e)
            print(
                "Did you use the right task attributes? Please refer to the online documentation: https://docs.segments.ai/reference/categories-and-task-attributes#object-attribute-format."
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

            # with open(
            #     f"/Users/arnouthillen/segments-ai/segments2/tests/add_dataset.json",
            #     "w",
            # ) as f:
            #     print(json.dump(r.json(), f))

            dataset = from_dict(data_class=Dataset, data=r.json(), config=DACITE_CONFIG)
            return dataset

    def update_dataset(
        self,
        dataset_identifier: str,
        description: Optional[str] = None,
        task_type: Optional[TaskType] = None,
        task_attributes: Optional[Dict[str, Any]] = None,
        category: Optional[str] = None,
        public: Optional[bool] = None,
        readme: Optional[str] = None,
        enable_skip_labeling: Optional[bool] = None,
        enable_skip_reviewing: Optional[bool] = None,
        enable_ratings: Optional[bool] = None,
    ) -> Dataset:
        """Update a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            description (str, optional): The dataset description.
            task_type (str, optional): The dataset's task type. One of 'segmentation-bitmap', 'segmentation-bitmap-highres', 'vector', 'bboxes', 'keypoints', 'pointcloud-segmentation', 'pointcloud-detection'.
            task_attributes (dict, optional): The dataset's task attributes.
            category (str, optional): The dataset category.
            public (bool, optional): The dataset visibility.
            readme (str, optional): The dataset readme.
            enable_skip_labeling (bool, optional): Enable the skip button in the labeling workflow.
            enable_skip_reviewing (bool, optional): Enable the skip button in the reviewing workflow.
            enable_ratings (bool, optional): Enable star-ratings for labeled images.

        Returns:
            dict: a dictionary representing the updated dataset.
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

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/update_dataset.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        print("Updated " + dataset_identifier)
        dataset = from_dict(data_class=Dataset, data=r.json(), config=DACITE_CONFIG)
        return dataset

    def delete_dataset(self, dataset_identifier: str) -> None:
        """Delete a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
        """

        r = self.delete("/datasets/{}/".format(dataset_identifier))

    def add_dataset_collaborator(
        self, dataset_identifier: str, username: str, role: Role = "labeler"
    ) -> Collaborator:
        """Add a collaborator to a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            username (str): The username of the collaborator to be added.
            role (str, optional): The role of the collaborator to be added. One of labeler, reviewer, admin. Defaults to labeler.

        Returns:
            dict: a dictionary containing the newly added collaborator with its role.
        """
        payload = {"user": username, "role": role}
        r = self.post("/datasets/{}/collaborators/".format(dataset_identifier), payload)

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/add_dataset_collaborator.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        collaborator = from_dict(
            data_class=Collaborator, data=r.json(), config=DACITE_CONFIG
        )
        return collaborator

    def delete_dataset_collaborator(
        self, dataset_identifier: str, username: str
    ) -> Collaborator:
        """Delete a dataset collaborator.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            username (str): The username of the collaborator to be deleted.
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
        metadata: Optional[Dict[str, Any]] = None,
        sort: Literal["name", "created", "priority"] = "name",
        direction: Literal["asc", "desc"] = "asc",
        per_page: int = 1000,
        page: int = 1,
    ) -> List[Sample]:
        """Get the samples in a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str, optional): Name to filter by. Defaults to None (no filtering).
            label_status (list, optional): List of label statuses to filter by. Defaults to None (no filtering).
            metadata (list, optional): List of 'key:value' metadata attributes to filter by. Defaults to None (no filtering).
            sort (str, optional): What to sort results by. One of 'name', 'created', 'priority'. Defaults to 'name'.
            direction (str, optional): Sorting direction. One of 'asc' (ascending) or 'desc' (descending). Defaults to 'asc'.
            per_page (int, optional): Pagination parameter indicating the maximum number of samples to return. Defaults to 1000.
            page (int, optional): Pagination parameter indicating the page to return. Defaults to 1.

        Returns:
            list: a list of dictionaries representing the samples.
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
            label_status = [status.upper() for status in label_status]
            query_string += "&labelset=ground-truth&label_status={}".format(
                ",".join(label_status)
            )

        # sorting
        sort_dict = {"name": "name", "created": "created_at", "priority": "priority"}
        assert sort in sort_dict
        assert direction in ["asc", "desc"]
        if sort != "name":
            direction_str = "" if direction == "asc" else "-"
            sort_str = sort_dict[sort]
            query_string += "&sort={}{}".format(direction_str, sort_str)

        r = self.get("/datasets/{}/samples/{}".format(dataset_identifier, query_string))
        results = r.json()

        for result in results:
            result.pop("label", None)

        # with open(
        #     "/Users/arnouthillen/segments-ai/segments2/tests/get_samples.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        results = [
            from_dict(data_class=Sample, data=result, config=DACITE_CONFIG)
            for result in results
        ]
        return results

    def get_sample(self, uuid: str, labelset: Optional[str] = None) -> Sample:
        """Get a sample.

        Args:
            uuid (str): The sample uuid.
            labelset (str, optional): If defined, this additionally returns the label for the given labelset. Defaults to None.

        Returns:
            dict: a dictionary representing the sample
        """

        query_string = "/samples/{}/".format(uuid)

        if labelset is not None:
            query_string += "?labelset={}".format(labelset)

        r = self.get(query_string)

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/get_sample_{index}.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        sample = from_dict(data_class=Sample, data=r.json(), config=DACITE_CONFIG)
        return sample

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

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the sample.
            attributes (dict): The sample attributes. Please refer to the online documentation.
            metadata (dict, optional): Any sample metadata. Example: {'weather': 'sunny', 'camera_id': 3}.
            priority (float, optional): Priority in the labeling queue. Samples with higher values will be labeled first. Defaults to 0.
            embedding (array, optional): Embedding of this sample represented by an array of floats.

        Returns:
            dict: a dictionary representing the newly created sample.
        """

        try:
            from_dict(
                data_class=SampleAttributes, data=attributes, config=DACITE_CONFIG
            )
        except Exception as e:
            print(e)
            print(
                "Did you use the right sample attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/sample-types."
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
            sample = from_dict(data_class=Sample, data=r.json(), config=DACITE_CONFIG)
            return sample

    def update_sample(
        self,
        uuid: str,
        attributes: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        priority: float = 0,
        embedding: Optional[List[float]] = None,
    ) -> Sample:
        """Update a sample.

        Args:
            uuid (str): The sample uuid.
            name (str, optional): The name of the sample.
            attributes (dict, optional): The sample attributes. Please refer to the online documentation.
            metadata (dict, optional): Any sample metadata. Example: {'weather': 'sunny', 'camera_id': 3}.
            priority (float, optional): Priority in the labeling queue. Samples with higher values will be labeled first. Default is 0.
            embedding (array, optional): Embedding of this sample represented by an array of floats.

        Returns:
            dict: a dictionary representing the updated sample.
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
        sample = from_dict(data_class=Sample, data=r.json(), config=DACITE_CONFIG)
        return sample

    def delete_sample(self, uuid: str) -> None:
        """Delete a sample.

        Args:
            uuid (str): The sample uuid.
        """

        r = self.delete("/samples/{}/".format(uuid))

    ##########
    # Labels #
    ##########
    def get_label(self, sample_uuid: str, labelset: str = "ground-truth") -> Label:
        """Get a label.

        Args:
            sample_uuid (str): The sample uuid.
            labelset (str, optional): The labelset this label belongs to. Defaults to 'ground-truth'.

        Returns:
            dict: a dictionary representing the label.
        """

        r = self.get("/labels/{}/{}/".format(sample_uuid, labelset))

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/get_label_{index}.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        label = from_dict(data_class=Label, data=r.json(), config=DACITE_CONFIG)
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
            sample_uuid (str): The sample uuid.
            labelset (str): The labelset this label belongs to.
            attributes (dict): The label attributes. Please refer to the online documentation.
            label_status (str, optional): The label status. Defaults to 'PRELABELED'.
            score (float, optional): The label score. Defaults to None.

        Returns:
            dict: a dictionary representing the newly created label.
        """

        try:
            from_dict(data_class=LabelAttributes, data=attributes, config=DACITE_CONFIG)
        except Exception as e:
            print(e)
            print(
                "Did you use the right label attributes? Please refer to the online documentation: https://docs.segments.ai/reference/sample-and-label-types/label-types."
            )
        else:
            payload: Dict[str, Any] = {
                "label_status": label_status,
                "attributes": attributes,
            }

            if score is not None:
                payload["score"] = score

            r = self.put("/labels/{}/{}/".format(sample_uuid, labelset), payload)
            label = from_dict(data_class=Label, data=r.json(), config=DACITE_CONFIG)
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
            sample_uuid (str): The sample uuid.
            labelset (str): The labelset this label belongs to.
            attributes (dict): The label attributes. Please refer to the online documentation.
            label_status (str, optional): The label status. Defaults to 'PRELABELED'.
            score (float, optional): The label score. Defaults to None.

        Returns:
            dict: a dictionary representing the updated label.
        """

        payload: Dict[str, Any] = {}

        if attributes is not None:
            payload["attributes"] = attributes

        if label_status is not None:
            payload["label_status"] = label_status

        if score is not None:
            payload["score"] = score

        r = self.patch("/labels/{}/{}/".format(sample_uuid, labelset), payload)
        label = from_dict(data_class=Label, data=r.json(), config=DACITE_CONFIG)
        return label

    def delete_label(self, sample_uuid: str, labelset: str) -> None:
        """Delete a label.

        Args:
            sample_uuid (str): The sample uuid.
            labelset (str): The labelset this label belongs to.
        """

        r = self.delete("/labels/{}/{}/".format(sample_uuid, labelset))

    #############
    # Labelsets #
    #############
    def get_labelsets(self, dataset_identifier: str) -> List[LabelSet]:
        """Get the labelsets in a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            list: a list of dictionaries representing the labelsets.
        """

        r = self.get("/datasets/{}/labelsets/".format(dataset_identifier))

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/get_labelsets.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        labelsets = [
            from_dict(data_class=LabelSet, data=labelset, config=DACITE_CONFIG)
            for labelset in r.json()
        ]
        return labelsets

    def get_labelset(self, dataset_identifier: str, name: str) -> LabelSet:
        """Get a labelset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the labelset.

        Returns:
            dict: a dictionary representing the labelset.
        """

        r = self.get("/datasets/{}/labelsets/{}/".format(dataset_identifier, name))

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/get_labelset_{i}.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        labelset = from_dict(data_class=LabelSet, data=r.json(), config=DACITE_CONFIG)
        return labelset

    def add_labelset(
        self, dataset_identifier: str, name: str, description: str = ""
    ) -> LabelSet:
        """Add a labelset to a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the labelset.
            description (str, optional): The labelset description.

        Returns:
            dict: a dictionary representing the labelset.
        """

        payload = {
            "name": name,
            "description": description,
            "attributes": "{}",
        }
        r = self.post("/datasets/{}/labelsets/".format(dataset_identifier), payload)

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/add_delete_labelset.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        labelset = from_dict(data_class=LabelSet, data=r.json(), config=DACITE_CONFIG)
        return labelset

    def delete_labelset(self, dataset_identifier: str, name: str) -> None:
        """Delete a labelset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the labelset.
        """

        r = self.delete("/datasets/{}/labelsets/{}/".format(dataset_identifier, name))

    ############
    # Releases #
    ############
    def get_releases(self, dataset_identifier: str) -> List[Release]:
        """Get the releases in a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.

        Returns:
            list: a list of dictionaries representing the releases.
        """

        r = self.get("/datasets/{}/releases/".format(dataset_identifier))

        # with open(
        #     "/Users/arnouthillen/segments-ai/segments2/tests/get_releases.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        releases = [
            from_dict(data_class=Release, data=release, config=DACITE_CONFIG)
            for release in r.json()
        ]
        return releases

    def get_release(self, dataset_identifier: str, name: str) -> Release:
        """Get a release.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the release.

        Returns:
            dict: a dictionary representing the release.
        """

        r = self.get("/datasets/{}/releases/{}/".format(dataset_identifier, name))

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/get_release_{index}.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        release = from_dict(data_class=Release, data=r.json(), config=DACITE_CONFIG)
        return release

    def add_release(
        self, dataset_identifier: str, name: str, description: str = ""
    ) -> Release:
        """Add a release to a dataset.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the release.
            description (str, optional): The release description.

        Returns:
            dict: a dictionary representing the newly created release.
        """

        payload = {"name": name, "description": description}
        r = self.post("/datasets/{}/releases/".format(dataset_identifier), payload)

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/add_delete_release.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        release = from_dict(data_class=Release, data=r.json(), config=DACITE_CONFIG)
        return release

    def delete_release(self, dataset_identifier: str, name: str) -> None:
        """Delete a release.

        Args:
            dataset_identifier (str): The dataset identifier, consisting of the name of the dataset owner followed by the name of the dataset itself. Example: jane/flowers.
            name (str): The name of the release.
        """

        r = self.delete("/datasets/{}/releases/{}/".format(dataset_identifier, name))

    ##########
    # Assets #
    ##########
    def upload_asset(self, file: IO, filename: str = "label.png") -> File:
        """Upload an asset.

        Args:
            file (object): A file object.
            filename (str, optional): The file name. Defaults to label.png.

        Returns:
            dict: a dictionary representing the uploaded file.
        """

        r = self.post("/assets/", {"filename": filename})
        response_aws = self._upload_to_aws(file, r.json()["presignedPostFields"])

        # with open(
        #     f"/Users/arnouthillen/segments-ai/segments2/tests/upload_asset.json",
        #     "w",
        # ) as f:
        #     print(json.dump(r.json(), f))

        file = from_dict(data_class=File, data=r.json(), config=DACITE_CONFIG)
        return file

    # Error handling: https://stackoverflow.com/questions/16511337/correct-way-to-try-except-using-python-requests-module

    def _get_auth_header(self) -> Optional[AuthHeader]:
        if self.api_key:
            return {"Authorization": "APIKey {}".format(self.api_key)}
        else:
            return None

    def get(self, endpoint: str, auth: bool = True) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.get(
                urllib.parse.urljoin(self.api_url, endpoint), headers=headers
            )
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{} | {}".format(errh, r.json()))

        return r

    def post(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, auth: bool = True
    ) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.post(
                urllib.parse.urljoin(self.api_url, endpoint),
                json=data,  # data=data
                headers=headers,
            )
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{} | {}".format(errh, r.json()))

        return r

    def put(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, auth: bool = True
    ) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.put(
                urllib.parse.urljoin(self.api_url, endpoint),
                json=data,  # data=data
                headers=headers,
            )
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{} | {}".format(errh, r.json()))

        return r

    def patch(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, auth: bool = True
    ) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.patch(
                urllib.parse.urljoin(self.api_url, endpoint),
                json=data,  # data=data
                headers=headers,
            )
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{} | {}".format(errh, r.json()))

        return r

    def delete(
        self, endpoint: str, data: Optional[Dict[str, Any]] = None, auth: bool = True
    ) -> requests.Response:
        headers = self._get_auth_header() if auth else None

        try:
            r = self.api_session.delete(
                urllib.parse.urljoin(self.api_url, endpoint),
                json=data,  # data=data
                headers=headers,
            )
            r.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("{}".format(errh))

        return r

    def _upload_to_aws(self, file: IO, aws_fields: AWSFields) -> requests.Response:
        files = {"file": file}
        r = self.s3_session.post(
            aws_fields["url"], files=files, data=aws_fields["fields"]
        )
        return r

    # https://stackoverflow.com/questions/48160728/resourcewarning-unclosed-socket-in-python-3-unit-test
    def close(self) -> None:
        self.api_session.close()
        self.s3_session.close()
        print("Closed successfully.")
