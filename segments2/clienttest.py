import json
import os
from unicodedata import name
from dotenv import load_dotenv
from client import SegmentsClient
from unittest import TestCase, main
from dacite import from_dict
from typehints import (
    DACITE_CONFIG,
    Collaborator,
    Dataset,
    File,
    Label,
    LabelSet,
    Release,
    Sample,
    ImageSampleAttributes,
    ImageSequenceSampleAttributes,
    PointcloudSampleAttributes,
    PointcloudSequenceSampleAttributes,
    TextSampleAttributes,
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

        # with open(self.path + "get_datasets.json", "r") as f:
        #     j = json.load(f)
        #     test_datasets = self.client.get_datasets()
        #     datasets = [
        #         from_dict(data_class=Dataset, data=dataset, config=DACITE_CONFIG)
        #         for dataset in j
        #     ]
        #     for test_dataset, dataset in zip(test_datasets, datasets):
        #         test_dataset.created_at = ""
        #         dataset.created_at = ""
        #         self.assertIsInstance(test_dataset, Dataset)
        #         self.assertEqual(test_dataset, dataset)

    def test_get_dataset(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        dataset = self.client.get_dataset(dataset_identifier)
        self.assertIsInstance(dataset, Dataset)

        # dataset_identifier = self.owner + "/" + self.datasets[0]
        # with open(self.path + "get_dataset.json", "r") as f:
        #     j = json.load(f)
        #     dataset = from_dict(data_class=Dataset, data=j, config=DACITE_CONFIG)
        #     test_dataset = self.client.get_dataset(dataset_identifier)
        #     self.assertIsInstance(test_dataset, Dataset)
        #     self.assertEqual(test_dataset, dataset)

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
        # try:
        #     # Add dataset
        #     with open(self.path + "add_dataset.json", "r") as f:
        #         j = json.load(f)
        #         dataset = from_dict(data_class=Dataset, data=j, config=DACITE_CONFIG)
        #         test_dataset = self.client.add_dataset(**arguments)
        #         self.assertIsInstance(test_dataset, Dataset)
        #         test_dataset.created_at = ""
        #         dataset.created_at = ""
        #         self.assertEqual(test_dataset, dataset)

        #     # Update dataset
        #     with open(self.path + "update_dataset.json", "r") as f:
        #         j = json.load(f)
        #         dataset = from_dict(data_class=Dataset, data=j, config=DACITE_CONFIG)
        #         arguments["dataset_identifier"] = self.owner + "/" + arguments["name"]
        #         del arguments["name"]
        #         test_dataset = self.client.update_dataset(**arguments)
        #         self.assertIsInstance(test_dataset, Dataset)
        #         test_dataset.created_at = ""
        #         dataset.created_at = ""
        #         self.assertEqual(test_dataset, dataset)

        # finally:
        #     # Delete dataset
        #     self.client.delete_dataset(self.owner + "/add_dataset")

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
        # try:
        #     with open(self.path + "add_delete_dataset_collaborator.json", "r") as f:
        #         j = json.load(f)
        #         collaborator = from_dict(
        #             data_class=Collaborator, data=j, config=DACITE_CONFIG
        #         )
        #         dataset_identifier = self.owner + "/" + self.datasets[0]
        #         username = "admin-arnout"
        #         role = "admin"
        #         test_collaborator = self.client.add_dataset_collaborator(
        #             dataset_identifier, username, role
        #         )
        #         self.assertIsInstance(test_collaborator, Collaborator)
        #         self.assertEqual(test_collaborator, collaborator)
        # finally:
        #     self.client.delete_dataset_collaborator(dataset_identifier, username)


# print(client.get_datasets())
# client.get_dataset("admin-arnout/example-text-span-categorization")
# client.add_dataset("sdk-test")
# client.update_dataset("admin-arnout/sdk-test")
# client.delete_dataset("segments-arnout/sdk-test")
# client.add_dataset_collaborator("segments-arnout/sdk-test", "arnouthillen")

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
        # with open(self.path + "get_samples.json", "r") as f:
        #     j = json.load(f)
        #     samples = [
        #         from_dict(data_class=Sample, data=sample, config=DACITE_CONFIG)
        #         for sample in j
        #     ]
        #     for test_sample, sample in zip(test_samples, samples):
        #         self.assertIsInstance(test_sample, Sample)
        #         self.assertEqual(test_sample, sample)

    def test_get_sample(self):
        labelset = "ground-truth"
        for i in range(len(self.uuids)):
            sample = self.client.get_sample(self.uuids[i], labelset)
            self.assertIsInstance(sample, Sample)
        # labelset = "ground-truth"
        # for i in range(len(self.uuids)):
        #     test_sample = self.client.get_sample(self.uuids[i], labelset)
        #     self.assertIsInstance(test_sample, Sample)
        #     with open(self.path + f"get_sample_{i}.json", "r") as f:
        #         j = json.load(f)
        #         sample = from_dict(data_class=Sample, data=j, config=DACITE_CONFIG)
        #         self.assertEqual(test_sample, sample)

    def test_add_update_delete_sample(self):

        dataset_identifier=self.owner+"/"
        metadata={}
        priority=0
        embedding=[]
        name=""
        try:
            for dataset in self.datasets:
            attributes =

        finally:



# pointcloud = "admin-arnout/example-point-clouds-segmentation"
# pointcloud_sequence =
# image =
# image_sequence =
# text =
# client.get_samples(pointcloud)
# client.get_sample("9b09321f-37d0-49cf-8d5a-ac752dc69672")
# client.add_sample()
# client.update_sample()
# client.delete_sample()

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
        # labelset = "ground-truth"
        # for i in range(len(self.uuids)):
        #     test_label = self.client.get_label(self.uuids[i], labelset)
        #     with open(self.path + f"get_label_{i}.json", "r") as f:
        #         j = json.load(f)
        #         label = from_dict(data_class=Label, data=j, config=DACITE_CONFIG)
        #         self.assertIsInstance(test_label, Label)
        #         self.assertEqual(test_label, label)

    def test_add_update_delete_label(self):
        pass


# text_uuid = "545bb6e7-0034-4455-9d12-48a8ddc6d919"
# pc_segmentation_uuid = "9b09321f-37d0-49cf-8d5a-ac752dc69672"
# pc_cuboid_uuid = "efe26b83-c27b-4514-b94e-62c9112ed043"
# pc_sequence_segmentation_uuid = "d1766c11-34d4-4f43-bb02-95f98a44c806"  # error
# pc_sequence_cuboid_uuid = "38b8c7d2-f682-473b-96ef-e66809033398"
# image_segmentation_uuid = "1b398e23-61ae-4bd1-a1b6-5f0a3e1c1d5a"
# image_vector_uuid = "f213294f-8131-4a07-981d-f3289a3eec04"
# image_sequence_vector_uuid = "1505f746-3a9a-449f-b1cf-c70160dbc06a"  # error
# print(client.get_label(image_sequence_vector_uuid))
# client.add_label()
# client.update_label()
# client.delete_label()

############
# Labelset #
############
class TestLabelSet(Test):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_get_labelsets(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        labelsets = self.client.get_labelsets(dataset_identifier)
        for labelset in labelsets:
            self.assertIsInstance(labelset, LabelSet)
        # dataset_identifier = self.owner + "/" + self.datasets[0]
        # test_labelsets = self.client.get_labelsets(dataset_identifier)
        # with open(self.path + f"get_labelsets.json", "r") as f:
        #     j = json.load(f)
        #     labelsets = [
        #         from_dict(data_class=LabelSet, data=labelset, config=DACITE_CONFIG)
        #         for labelset in j
        #     ]
        #     test_labelsets = self.client.get_labelsets(dataset_identifier)
        #     for labelset, test_labelset in zip(labelsets, test_labelsets):
        #         self.assertIsInstance(test_labelset, LabelSet)
        #         self.assertEqual(test_labelset, labelset)

    def test_get_labelset(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        for i in range(len(self.labelsets)):
            labelset = self.client.get_labelset(dataset_identifier, self.labelsets[i])
            self.assertIsInstance(labelset, LabelSet)
        # dataset_identifier = self.owner + "/" + self.datasets[0]
        # for i in range(len(self.labelsets)):
        #     test_labelset = self.client.get_labelset(
        #         dataset_identifier, self.labelsets[i]
        #     )
        #     with open(self.path + f"get_labelset_{i}.json", "r") as f:
        #         j = json.load(f)
        #         labelset = from_dict(data_class=LabelSet, data=j, config=DACITE_CONFIG)
        #         self.assertIsInstance(test_labelset, LabelSet)
        #         self.assertEqual(test_labelset, labelset)

    def test_add_delete_labelset(self):
        # Add labelset.
        try:
            dataset_identifier = self.owner + "/" + self.datasets[0]
            name = "labelset4"
            description = "Test add_delete_labelset description."
            labelset = self.client.add_labelset(dataset_identifier, name, description)
            self.assertIsInstance(labelset, LabelSet)
        # Delete labelset.
        finally:
            self.client.delete_labelset(dataset_identifier, name)
        # # Add labelset.
        # try:
        #     dataset_identifier = self.owner + "/" + self.datasets[0]
        #     name = "labelset4"
        #     description = "Test add_delete_labelset description."
        #     test_labelset = self.client.add_labelset(
        #         dataset_identifier, name, description
        #     )
        #     with open(self.path + f"add_delete_labelset.json", "r") as f:
        #         j = json.load(f)
        #         labelset = from_dict(data_class=LabelSet, data=j, config=DACITE_CONFIG)
        #         self.assertIsInstance(test_labelset, LabelSet)
        #         test_labelset.uuid = ""
        #         test_labelset.created_at = ""
        #         labelset.uuid = ""
        #         labelset.created_at = ""
        #         self.assertEqual(test_labelset, labelset)

        # # Delete labelset.
        # finally:
        #     self.client.delete_labelset(dataset_identifier, name)


# name = "admin-arnout/example-text-span-categorization"
# print(client.get_labelsets(name))
# print(client.get_labelset(name, "ground-truth"))
# print(client.add_labelset(name, "test5", "test"))
# client.delete_labelset(name, "test5")

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
        # dataset_identifier = self.owner + "/" + self.datasets[0]
        # with open(self.path + "get_releases.json", "r") as f:
        #     j = json.load(f)
        #     test_releases = self.client.get_releases(dataset_identifier)
        #     releases = [
        #         from_dict(data_class=Release, data=release, config=DACITE_CONFIG)
        #         for release in j
        #     ]
        #     for test_release, release in zip(test_releases, releases):
        #         self.assertIsInstance(test_release, Release)
        #         self.assertEqual(test_release, release)

    def test_get_release(self):
        dataset_identifier = self.owner + "/" + self.datasets[0]
        for i in range(len(self.releases)):
            release = self.client.get_release(dataset_identifier, self.releases[i])
            self.assertIsInstance(release, Release)
        # dataset_identifier = self.owner + "/" + self.datasets[0]
        # for i in range(len(self.releases)):
        #     with open(self.path + f"get_release_{i}.json", "r") as f:
        #         j = json.load(f)
        #         release = from_dict(data_class=Release, data=j, config=DACITE_CONFIG)
        #         test_release = self.client.get_release(
        #             dataset_identifier, self.releases[i]
        #         )
        #         self.assertIsInstance(test_release, Release)
        #         self.assertEqual(test_release, release)

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

        # dataset_identifier = self.owner + "/" + self.datasets[0]
        # name = "v0.4"
        # description = "Test release description."

        # # Add release
        # try:
        #     with open(self.path + "add_delete_release.json", "r") as f:
        #         j = json.load(f)
        #         test_release = self.client.add_release(
        #             dataset_identifier, name, description
        #         )
        #         release = from_dict(data_class=Release, data=j, config=DACITE_CONFIG)
        #         self.assertIsInstance(test_release, Release)
        #         test_release.created_at = ""
        #         test_release.uuid = ""
        #         release.created_at = ""
        #         release.uuid = ""
        #         self.assertEqual(test_release, release)
        # # Delete release
        # finally:
        #     self.client.delete_release(dataset_identifier, name)


# name = "admin-arnout/example-text-span-categorization"
# release = "v3"
# client.get_releases(name)
# print(client.get_release(name, release))
# print(client.add_release(name, release))
# client.delete_release(name, release)  # deletes release but throws error

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
            # with open(self.path + "upload_asset.json") as f:
            #     j = json.load(f)
            #     file = from_dict(data_class=File, data=j, config=DACITE_CONFIG)
            self.assertIsInstance(test_file, File)
            #     self.assertEqual(test_file, file)


if __name__ == "__main__":
    main()
