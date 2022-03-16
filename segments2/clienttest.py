import json
import os
from unicodedata import name
from dotenv import load_dotenv
from client import SegmentsClient
from unittest import TestCase, main
from typehints import (
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
class Test(TestCase):
    def setUp(self):
        ENV_PATH = "/Users/arnouthillen/segments-ai/segments2/.env"
        # https://pypi.org/project/python-dotenv/
        load_dotenv(dotenv_path=ENV_PATH)
        API_KEY = os.getenv("API_KEY")
        self.owner = os.getenv("OWNER")
        self.path = os.getenv("TESTS_PATH")
        self.client = SegmentsClient(api_key=API_KEY)
        self.datasets = json.loads(os.getenv("DATASETS"))
        # First sample uuid.
        self.uuids = json.loads(os.getenv("UUIDS"))
        # First sample of first dataset.
        self.labelsets = json.loads(os.getenv("LABELSETS"))
        # Releases of first dataset.
        self.releases = json.loads(os.getenv("RELEASES"))
        # Sample attribute type of the datasets.
        self.sample_attributes = json.loads(os.getenv("SAMPLE_ATTRIBUTES"))
        # Label attribute type of the datasets.
        self.label_attributes = json.loads(os.getenv("LABEL_ATTRIBUTES"))

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
        try:
            dataset_identifier = self.owner + "/" + self.datasets[0]
            username = "admin-arnout"
            role = "admin"
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
        for i in range(len(self.uuids)):
            sample = self.client.get_sample(self.uuids[i], labelset)
            self.assertIsInstance(sample, Sample)

    def test_add_update_delete_sample(self):
        dataset_identifier = self.owner + "/"
        metadata = {}
        priority = 0
        embedding = []
        name = ""
        # try:
        #     for dataset in self.datasets:
        #     attributes =
        # finally:


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
        for i in range(len(self.uuids)):
            label = self.client.get_label(self.uuids[i], labelset)
            self.assertIsInstance(label, Label)

    def test_add_update_delete_label(self):
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
        try:
            # Add labelset.
            dataset_identifier = self.owner + "/" + self.datasets[0]
            name = "labelset4"
            description = "Test add_delete_labelset description."
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
        with open(self.path + "test.png", "rb") as f:
            test_file = self.client.upload_asset(f, "test.png")
            self.assertIsInstance(test_file, File)


if __name__ == "__main__":
    main()
