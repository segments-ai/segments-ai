import json
import os
import unittest
import numpy as np
from segments.client import SegmentsClient
from segments.typehints import (
    Collaborator,
    Dataset,
    File,
    Label,
    Labelset,
    Release,
    Sample,
)

##############
# Base class #
##############
class Test(unittest.TestCase):
    def setUp(self):
        API_KEY = os.getenv("API_KEY", "")
        self.owner = os.getenv("OWNER", "")
        self.path = "tests"
        self.client = SegmentsClient(api_key=API_KEY)
        self.datasets = json.loads(os.getenv("DATASETS", ""))
        # First sample uuid.
        self.sample_uuids = json.loads(os.getenv("SAMPLE_UUIDS", ""))
        # First sample of first dataset.
        self.labelsets = json.loads(os.getenv("LABELSETS", ""))
        # Releases of first dataset.
        self.releases = json.loads(os.getenv("RELEASES", ""))
        # Sample attribute type of the datasets.
        self.sample_attributes = json.loads(os.getenv("SAMPLE_ATTRIBUTES", ""))
        # Label attribute type of the datasets.
        self.label_attributes = json.loads(os.getenv("LABEL_ATTRIBUTES", ""))

    def tearDown(self):
        self.client.close()


###########
# Dataset #
###########
class TestDataset(Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_get_datasets(self):
        datasets = self.client.get_datasets()
        for dataset in datasets:
            self.assertIsInstance(dataset, Dataset)

    def test_get_dataset(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        dataset = self.client.get_dataset(dataset_identifier)
        self.assertIsInstance(dataset, Dataset)

    def test_add_update_delete_dataset(self):
        arguments = {
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

    def test_add_delete_dataset_collaborator(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        username = "admin-arnout"
        role = "admin"
        try:
            collaborator = self.client.add_dataset_collaborator(
                dataset_identifier, username, role
            )
            self.assertIsInstance(collaborator, Collaborator)
        finally:
            self.client.delete_dataset_collaborator(dataset_identifier, username)


##########
# Sample #
##########
class TestSample(Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_get_samples(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        name = None
        label_status = None
        metadata = None
        sort = "created"
        direction = "desc"
        samples = self.client.get_samples(
            dataset_identifier, name, label_status, metadata, sort, direction
        )
        for sample in samples:
            self.assertIsInstance(sample, Sample)

    def test_get_sample(self):
        labelset = "ground-truth"
        for i in range(len(self.sample_uuids)):
            sample = self.client.get_sample(self.sample_uuids[i], labelset)
            self.assertIsInstance(sample, Sample)

    def test_add_update_delete_sample(self):
        metadata = {"weather": "sunny", "camera_id": 3}
        priority = 0
        # embedding = np.zeros(100).tolist()
        name = "Test sample"
        attributes_dict = {
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
        for sample_attribute, dataset in zip(self.sample_attributes, self.datasets):
            dataset_identifier = self.owner + "/" + dataset
            attributes = attributes_dict[sample_attribute]
            try:
                sample = self.client.add_sample(
                    dataset_identifier,
                    name,
                    attributes,
                    metadata,
                    priority,  # embedding
                )
                sample = self.client.update_sample(
                    sample.uuid,
                    name,
                    attributes,
                    metadata,
                    priority,  # embedding
                )
            finally:
                self.client.delete_sample(sample.uuid)


#########
# Label #
#########
class TestLabel(Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_get_label(self):
        labelset = "ground-truth"
        for i in range(len(self.sample_uuids)):
            label = self.client.get_label(self.sample_uuids[i], labelset)
            self.assertIsInstance(label, Label)

    def test_add_update_delete_label(self):
        label_attributes = {}
        labelset = "ground-truth"
        label_status = "PRELABELED"
        score = 0
        for sample_uuid in self.sample_uuids:
            pass


############
# Labelset #
############
class TestLabelset(Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_get_labelsets(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        labelsets = self.client.get_labelsets(dataset_identifier)
        for labelset in labelsets:
            self.assertIsInstance(labelset, Labelset)

    def test_get_labelset(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        for i in range(len(self.labelsets)):
            labelset = self.client.get_labelset(dataset_identifier, self.labelsets[i])
            self.assertIsInstance(labelset, Labelset)

    def test_add_delete_labelset(self):
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


###########
# Release #
###########
class TestRelease(Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_get_releases(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        releases = self.client.get_releases(dataset_identifier)
        for release in releases:
            self.assertIsInstance(release, Release)

    def test_get_release(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        for i in range(len(self.releases)):
            release = self.client.get_release(dataset_identifier, self.releases[i])
            self.assertIsInstance(release, Release)

    def test_add_delete_release(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        name = "v0.4"
        description = "Test release description."
        try:
            # Add release
            release = self.client.add_release(dataset_identifier, name, description)
            self.assertIsInstance(release, Release)
        finally:
            # Delete release
            self.client.delete_release(dataset_identifier, name)


#########
# Asset #
#########
class TestAsset(Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_upload_asset(self):
        with open(self.path + "/fixtures/test.png", "rb") as f:
            test_file = self.client.upload_asset(f, "test.png")
            self.assertIsInstance(test_file, File)


if __name__ == "__main__":
    unittest.main()
