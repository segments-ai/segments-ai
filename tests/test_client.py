from __future__ import annotations

import json
import os
import time
import unittest
from typing import Any, Dict

from segments.client import SegmentsClient
from segments.exceptions import NetworkError, ValidationError
from segments.typing import (
    Collaborator,
    Dataset,
    File,
    Label,
    Labelset,
    Release,
    Sample,
)
from typing_extensions import Final


##############
# Base class #
##############
class Test(unittest.TestCase):
    def setUp(self) -> None:
        API_KEY = os.getenv("SEGMENTS_API_KEY")
        API_URL = os.getenv("SEGMENTS_API_URL")
        self.owner = os.getenv("DATASET_OWNER", "")
        self.client = (
            SegmentsClient(api_key=API_KEY, api_url=API_URL)
            if API_URL is not None
            else SegmentsClient(api_key=API_KEY)
        )
        self.datasets = json.loads(os.getenv("DATASETS", ""))
        # First sample uuid.
        self.sample_uuids = json.loads(os.getenv("SAMPLE_UUIDS", ""))
        # First sample of first dataset.
        self.labelsets = json.loads(os.getenv("LABELSETS", ""))
        # Releases of first dataset.
        self.releases = json.loads(os.getenv("RELEASES", ""))
        # Sample attribute type of the datasets.
        self.sample_attribute_types = json.loads(
            os.getenv("SAMPLE_ATTRIBUTE_TYPES", "")
        )
        # Label attribute type of the datasets.
        self.label_attribute_types = json.loads(os.getenv("LABEL_ATTRIBUTE_TYPES", ""))
        self.TIME_INTERVAL = 0.2  # Wait for API call to complete.

    def tearDown(self) -> None:
        self.client._close()


###########
# Dataset #
###########
class TestDataset(Test):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_get_datasets(self) -> None:
        datasets = self.client.get_datasets()
        for dataset in datasets:
            self.assertIsInstance(dataset, Dataset)

    def test_get_dataset(self) -> None:
        dataset_identifier = self.owner + "/" + self.datasets[0]
        dataset = self.client.get_dataset(dataset_identifier)
        self.assertIsInstance(dataset, Dataset)

    def test_get_dataset_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            self.client.get_dataset(wrong_dataset_identifier)

    def test_add_update_delete_dataset(self) -> None:
        arguments: Dict[str, Any] = {
            "name": "add_dataset",
            "description": "Test add_update_delete_dataset.",
            "task_type": "vector",
            "task_attributes": {
                "format_version": "0.1",
                "categories": [
                    {
                        "name": "test",
                        "id": 1,
                        "color": [0, 0, 0, 0],
                        "attributes": [
                            {
                                "name": "color",
                                "input_type": "select",
                                "values": ["green", "yellow", "red"],
                                "default_value": "red",
                            },
                            {
                                "name": "description",
                                "input_type": "text",
                                "default_value": "A nice car.",
                            },
                            {
                                "name": "number_of_wheels",
                                "input_type": "number",
                                "min": 1,
                                "max": 20,
                                "step": 1,
                                "default_value": 4,
                            },
                            {
                                "name": "is_electric",
                                "input_type": "checkbox",
                                "default_value": False,
                            },
                        ],
                    },
                ],
            },
            "category": "people",
            "public": False,
            "readme": "Readme for add_update_delete_dataset.",
            "enable_skip_labeling": True,
            "enable_skip_reviewing": True,
            "enable_ratings": True,
        }
        try:
            # Add dataset
            dataset = self.client.add_dataset(**arguments)
            self.assertIsInstance(dataset, Dataset)

            # Update dataset
            arguments["dataset_identifier"] = self.owner + "/" + arguments["name"]
            del arguments["name"]
            dataset = self.client.update_dataset(**arguments)
            self.assertIsInstance(dataset, Dataset)

        finally:
            # Delete dataset
            self.client.delete_dataset(self.owner + "/add_dataset")

    def test_update_dataset_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            self.client.update_dataset(wrong_dataset_identifier)

    def test_delete_dataset_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            self.client.delete_dataset(wrong_dataset_identifier)

    def test_add_delete_dataset_collaborator(self) -> None:
        dataset_identifier = self.owner + "/" + self.datasets[0]
        username = "admin-arnout"
        role: Final = "admin"
        try:
            collaborator = self.client.add_dataset_collaborator(
                dataset_identifier, username, role
            )
            self.assertIsInstance(collaborator, Collaborator)
        finally:
            self.client.delete_dataset_collaborator(dataset_identifier, username)

    def test_delete_dataset_collaborator_networkerror(self) -> None:
        # Wrong dataset identifier and wrong username
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_username = "abcde"
            self.client.delete_dataset_collaborator(
                wrong_dataset_identifier, wrong_username
            )
        # Right dataset identifier and wrong username
        with self.assertRaises(NetworkError):
            right_dataset_identifier = self.owner + "/" + self.datasets[0]
            wrong_username = "abcde"
            self.client.delete_dataset_collaborator(
                right_dataset_identifier, wrong_username
            )


##########
# Sample #
##########
class TestSample(Test):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_get_samples(self) -> None:
        dataset_identifier = self.owner + "/" + self.datasets[0]
        name = None
        label_status = None
        metadata = None
        sort: Final = "created"
        direction: Final = "desc"
        samples = self.client.get_samples(
            dataset_identifier, name, label_status, metadata, sort, direction
        )
        for sample in samples:
            self.assertIsInstance(sample, Sample)

    def test_get_samples_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            self.client.get_samples(wrong_dataset_identifier)

    def test_get_sample(self) -> None:
        labelset = "ground-truth"
        for i in range(len(self.sample_uuids)):
            sample = self.client.get_sample(self.sample_uuids[i], labelset)
            self.assertIsInstance(sample, Sample)

    def test_get_sample_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_uuid = "12345"
            self.client.get_sample(wrong_uuid)

    def test_add_update_delete_sample(self) -> None:
        metadata = {"weather": "sunny", "camera_id": 3}
        priority = 0
        # embedding = np.zeros(100).tolist()
        name = "Test sample"
        attributes_dict: Dict[str, Dict[str, Any]] = {
            "image": {"image": {"url": "url"}},
            "image-sequence": {
                "frames": [{"image": {"url": "url"}}, {"image": {"url": ""}}]
            },
            "pointcloud": {
                "pcd": {"url": "url", "type": "kitti"},
                "ego_pose": {
                    "position": {"x": 0, "y": 0, "z": 0},
                    "heading": {"qx": 0, "qy": 0, "qz": 0, "qw": 0},
                },
                "default_z": -1,
                "name": "test_name",
                "timestamp": 1000,
            },
            "pointcloud-sequence": {
                "frames": [
                    {
                        "pcd": {"url": "url", "type": "kitti"},
                        "ego_pose": {
                            "position": {"x": 0, "y": 0, "z": 0},
                            "heading": {"qx": 0, "qy": 0, "qz": 0, "qw": 0},
                        },
                        "default_z": -1,
                        "name": "test_name",
                        "timestamp": 1000,
                    },
                    {
                        "pcd": {"url": "url", "type": "kitti"},
                        "ego_pose": {
                            "position": {"x": 0, "y": 0, "z": 0},
                            "heading": {"qx": 0, "qy": 0, "qz": 0, "qw": 0},
                        },
                        "default_z": -1,
                        "name": "test_name",
                        "timestamp": 1000,
                    },
                ]
            },
            "text": {"text": "This is a test sentence."},
        }
        for sample_attribute_type, dataset in zip(
            self.sample_attribute_types, self.datasets
        ):
            dataset_identifier = self.owner + "/" + dataset
            attributes = attributes_dict[sample_attribute_type]
            try:
                sample = self.client.add_sample(
                    dataset_identifier,
                    name,
                    attributes,
                    metadata,
                    priority,  # embedding
                )
                self.assertIsInstance(sample, Sample)
                sample = self.client.update_sample(
                    sample.uuid,
                    name,
                    attributes,
                    metadata,
                    priority,  # embedding
                )
                self.assertIsInstance(sample, Sample)
            finally:
                self.client.delete_sample(sample.uuid)

    def test_update_sample_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_uuid = "12345"
            self.client.update_sample(wrong_uuid)

    def test_delete_sample_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_uuid = "12345"
            self.client.delete_sample(wrong_uuid)


#########
# Label #
#########
class TestLabel(Test):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    # def test_get_label(self) -> None:
    #     labelset = "ground-truth"
    #     for sample_uuid in self.sample_uuids:
    #         label = self.client.get_label(sample_uuid, labelset)
    #         self.assertIsInstance(label, Label)

    def test_add_update_get_delete_label(self) -> None:
        # task_attributes = [
        #     {
        #         "name": "color",
        #         "input_type": "select",
        #         "values": ["green", "yellow", "red"],
        #         "default_value": "red",
        #     },
        #     {
        #         "name": "description",
        #         "input_type": "text",
        #         "default_value": "A nice car.",
        #     },
        #     {
        #         "name": "number_of_wheels",
        #         "input_type": "number",
        #         "min": "1",
        #         "max": "20",
        #         "step": "1",
        #         "default_value": 4,
        #     },
        #     {
        #         "name": "is_electric",
        #         "input_type": "checkbox",
        #         "default_value": False,
        #     },
        # ]
        image_or_object_attributes = (
            {  # sample-level attributes
                "scene_type": "crossroads",
                "weather": "sunny",
            },
        )
        label_attributes: Dict[str, Dict[str, Any]] = {
            "image-segmentation": {
                "format_version": "0.1",
                "annotations": [
                    {
                        "id": 1,
                        "category_id": 1,
                        "attributes": image_or_object_attributes,
                    },
                    {
                        "id": 2,
                        "category_id": 1,
                        "attributes": image_or_object_attributes,
                    },
                    {
                        "id": 3,
                        "category_id": 4,
                        "attributes": image_or_object_attributes,
                    },
                ],
                "segmentation_bitmap": {
                    "url": "https://segmentsai-staging.s3.eu-west-2.amazonaws.com/assets/davy/ddf55e99-1a6f-42d2-83e9-8657de3259a1.png"
                },
                "image_attributes": image_or_object_attributes,
            },
            "image-vector": {
                "format_version": "0.1",
                "annotations": [
                    {
                        "id": 1,
                        "category_id": 1,
                        "type": "bbox",
                        "points": [[12.34, 56.78], [90.12, 34.56]],
                        "attributes": image_or_object_attributes,
                    },
                    {
                        "id": 2,
                        "category_id": 2,
                        "type": "polygon",
                        "points": [
                            [12.34, 56.78],
                            [90.12, 34.56],
                            [78.91, 23.45],
                            [67.89, 98.76],
                            [54.32, 10.01],
                        ],
                        "attributes": image_or_object_attributes,
                    },
                    {
                        "id": 3,
                        "category_id": 3,
                        "type": "polyline",
                        "points": [
                            [12.34, 56.78],
                            [90.12, 34.56],
                            [78.91, 23.45],
                            [67.89, 98.76],
                            [54.32, 10.01],
                        ],
                        "attributes": image_or_object_attributes,
                    },
                    {
                        "id": 4,
                        "category_id": 4,
                        "type": "point",
                        "points": [[12.34, 56.78]],
                        "attributes": image_or_object_attributes,
                    },
                ],
                "image_attributes": image_or_object_attributes,
            },
            "image-sequence-vector": {
                "format_version": "0.2",
                "frames": [
                    {
                        "format_version": "0.1",
                        "timestamp": "00001",
                        "annotations": [
                            {
                                "id": 1,
                                "category_id": 1,
                                "track_id": 6,
                                "is_keyframe": True,
                                "type": "bbox",
                                "points": [[12.34, 56.78], [90.12, 34.56]],
                                "attributes": image_or_object_attributes,
                            },
                            {
                                "id": 2,
                                "category_id": 2,
                                "track_id": 5,
                                "is_keyframe": True,
                                "type": "polygon",
                                "points": [
                                    [12.34, 56.78],
                                    [90.12, 34.56],
                                    [78.91, 23.45],
                                    [67.89, 98.76],
                                    [54.32, 10.01],
                                ],
                                "attributes": image_or_object_attributes,
                            },
                            {
                                "id": 3,
                                "category_id": 3,
                                "track_id": 4,
                                "is_keyframe": True,
                                "type": "polyline",
                                "points": [
                                    [12.34, 56.78],
                                    [90.12, 34.56],
                                    [78.91, 23.45],
                                    [67.89, 98.76],
                                    [54.32, 10.01],
                                ],
                                "attributes": image_or_object_attributes,
                            },
                            {
                                "id": 4,
                                "category_id": 4,
                                "track_id": 3,
                                "is_keyframe": True,
                                "type": "point",
                                "points": [[12.34, 56.78]],
                                "attributes": image_or_object_attributes,
                            },
                        ],
                        "image_attributes": image_or_object_attributes,
                    }
                ],
            },
            "pointcloud-segmentation": {
                "format_version": "0.1",
                "annotations": [
                    {
                        "id": 1,
                        "category_id": 1,
                        "attributes": image_or_object_attributes,
                    },
                    {
                        "id": 2,
                        "category_id": 1,
                        "attributes": image_or_object_attributes,
                    },
                    {
                        "id": 3,
                        "category_id": 4,
                        "attributes": image_or_object_attributes,
                    },
                ],
                "point_annotations": [0, 0, 0, 3, 2, 2, 2, 1, 3],
            },
            "pointcloud-cuboid": {
                "format_version": "0.1",
                "annotations": [
                    {
                        "id": 1,
                        "category_id": 1,
                        "type": "cuboid",
                        "position": {"x": 0.0, "y": 0.2, "z": 0.5},
                        "dimensions": {"x": 1.2, "y": 1, "z": 1},
                        "yaw": 1.63,
                    }
                ],
            },
            "pointcloud-sequence-segmentation": {
                "format_version": "0.2",
                "frames": [
                    {
                        "format_version": "0.2",
                        "annotations": [
                            {
                                "id": 1,  # the object id
                                "category_id": 1,  # the category id
                                "track_id": 3,  # this id is used to link objects across frames
                            },
                            {"id": 2, "category_id": 1, "track_id": 4},
                            {"id": 3, "category_id": 4, "track_id": 5},
                        ],
                        "point_annotations": [
                            0,
                            0,
                            0,
                            3,
                            2,
                            2,
                            2,
                            1,
                            3,
                        ],  # refers to object ids
                    }
                ],
            },
            "pointcloud-sequence-cuboid": {
                "format_version": "0.2",
                "frames": [
                    {
                        "format_version": "0.2",
                        "timestamp": "00001",  # this field is only included if the sample has a timestamp
                        "annotations": [
                            {
                                "id": 1,  # the object id
                                "category_id": 1,  # the category id
                                "type": "cuboid",  # refers to the annotation type (cuboid)
                                "position": {"x": 0.0, "y": 0.2, "z": 0.5},
                                "dimensions": {"x": 1.2, "y": 1, "z": 1},
                                "yaw": 1.63,
                                "is_keyframe": True,  # whether this frame is a keyframe
                                "track_id": 6,  # this id is used to links objects across frames
                            },
                            {
                                "id": 2,
                                "category_id": 2,
                                "type": "cuboid",
                                "position": {"x": 0.0, "y": 0.2, "z": 0.5},
                                "dimensions": {"x": 1.2, "y": 1, "z": 1},
                                "yaw": 1.63,
                                "is_keyframe": False,
                                "track_id": 7,
                            },
                        ],
                    },
                ],
            },
            "text": {
                "format_version": "0.1",
                "annotations": [
                    {
                        "start": 0,  # the first character index of the label
                        "end": 5,  # the last character index of the the label (exclusive)
                        "category_id": 1,  # the category id
                    },
                    {"start": 7, "end": 12, "category_id": 0},
                    {"start": 20, "end": 30, "category_id": 2},
                ],
            },
        }
        labelset = "ground-truth"
        label_status: Final = "PRELABELED"
        score = 1
        for sample_uuid, label_attribute_type in zip(
            self.sample_uuids, self.label_attribute_types
        ):
            attributes = label_attributes[label_attribute_type]
            try:
                # Add
                label = self.client.add_label(
                    sample_uuid, attributes, labelset, label_status, score
                )
                self.assertIsInstance(label, Label)
                # Update
                label = self.client.update_label(
                    sample_uuid, attributes, labelset, label_status, score
                )
                self.assertIsInstance(label, Label)
                # Get
                label = self.client.get_label(sample_uuid, labelset)
                self.assertIsInstance(label, Label)
            finally:
                # Delete
                time.sleep(self.TIME_INTERVAL)
                self.client.delete_label(sample_uuid, labelset)

    def test_add_label_validationerror(self) -> None:
        with self.assertRaises(ValidationError):
            wrong_sample_uuid = "12345"
            wrong_attributes: Dict[str, Any] = {}
            self.client.add_label(wrong_sample_uuid, wrong_attributes)

    def test_update_label_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_sample_uuid = "12345"
            wrong_attributes: Dict[str, Any] = {}
            self.client.update_label(wrong_sample_uuid, wrong_attributes)


############
# Labelset #
############
class TestLabelset(Test):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_get_labelsets(self) -> None:
        dataset_identifier = self.owner + "/" + self.datasets[0]
        labelsets = self.client.get_labelsets(dataset_identifier)
        for labelset in labelsets:
            self.assertIsInstance(labelset, Labelset)

    def test_get_labelsets_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            self.client.get_labelsets(wrong_dataset_identifier)

    def test_get_labelset(self) -> None:
        dataset_identifier = self.owner + "/" + self.datasets[0]
        for i in range(len(self.labelsets)):
            labelset = self.client.get_labelset(dataset_identifier, self.labelsets[i])
            self.assertIsInstance(labelset, Labelset)

    def test_get_labelset_networkerror(self) -> None:
        # Wrong dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.get_labelset(wrong_dataset_identifier, wrong_name)
        # Right dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            right_dataset_identifier = self.owner + "/" + self.datasets[0]
            wrong_name = "abcde"
            self.client.get_labelset(right_dataset_identifier, wrong_name)

    def test_add_delete_labelset(self) -> None:
        # Add labelset.
        dataset_identifier = self.owner + "/" + self.datasets[0]
        name = "labelset4"
        description = "Test add_delete_labelset description."
        try:
            labelset = self.client.add_labelset(dataset_identifier, name, description)
            self.assertIsInstance(labelset, Labelset)
        finally:
            # Delete labelset.
            self.client.delete_labelset(dataset_identifier, name)

    def test_add_labelset_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.add_labelset(wrong_dataset_identifier, wrong_name)

    def test_delete_labelset_networkerror(self) -> None:
        # Wrong dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.delete_labelset(wrong_dataset_identifier, wrong_name)
        # Right dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            right_dataset_identifier = self.owner + "/" + self.datasets[0]
            wrong_name = "abcde"
            self.client.delete_labelset(right_dataset_identifier, wrong_name)


###########
# Release #
###########
class TestRelease(Test):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_get_releases(self) -> None:
        dataset_identifier = self.owner + "/" + self.datasets[0]
        releases = self.client.get_releases(dataset_identifier)
        for release in releases:
            self.assertIsInstance(release, Release)

    def test_get_releases_networkerror(self) -> None:
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            self.client.get_releases(wrong_dataset_identifier)

    def test_get_release(self) -> None:
        dataset_identifier = self.owner + "/" + self.datasets[0]
        for i in range(len(self.releases)):
            release = self.client.get_release(dataset_identifier, self.releases[i])
            self.assertIsInstance(release, Release)

    def test_get_release_networkerror(self) -> None:
        # Wrong dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.get_release(wrong_dataset_identifier, wrong_name)
        # Right dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            right_dataset_identifier = self.owner + "/" + self.datasets[0]
            wrong_name = "abcde"
            self.client.get_release(right_dataset_identifier, wrong_name)

    def test_add_delete_release(self) -> None:
        dataset_identifier = self.owner + "/" + self.datasets[0]
        name = "v0.4"
        description = "Test release description."
        try:
            # Add release
            release = self.client.add_release(dataset_identifier, name, description)
            self.assertIsInstance(release, Release)
        finally:
            # Delete release
            time.sleep(self.TIME_INTERVAL)
            self.client.delete_release(dataset_identifier, name)

    def test_add_release_networkerror(self) -> None:
        # Wrong dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.add_release(wrong_dataset_identifier, wrong_name)

    def test_delete_release_networkerror(self) -> None:
        # Wrong dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.delete_release(wrong_dataset_identifier, wrong_name)
        # Right dataset identifier and wrong name
        with self.assertRaises(NetworkError):
            right_dataset_identifier = self.owner + "/" + self.datasets[0]
            wrong_name = "abcde"
            self.client.delete_release(right_dataset_identifier, wrong_name)


#########
# Asset #
#########
class TestAsset(Test):
    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_upload_asset(self) -> None:
        with open("tests/fixtures/test.png", "rb") as f:
            test_file = self.client.upload_asset(f, "test.png")
            self.assertIsInstance(test_file, File)


if __name__ == "__main__":
    unittest.main()
