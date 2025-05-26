from __future__ import annotations

import time
import unittest
from typing import Any, Dict

import pytest
from segments.client import SegmentsClient
from segments.exceptions import (
    AlreadyExistsError,
    AuthenticationError,
    AuthorizationError,
    NetworkError,
    NotFoundError,
    ValidationError,
)
from segments.typing import (
    Collaborator,
    Dataset,
    File,
    Issue,
    Label,
    Labelset,
    LabelStatus,
    Release,
    Role,
    Sample,
    TaskType,
    User,
)
from typing_extensions import Final


########
# User #
########
@pytest.mark.usefixtures("setup_class_client")
class TestUser:
    def test_get_user(self, owner) -> None:
        # authenticated user
        user = self.client.get_user()
        assert isinstance(user, User)
        # other user
        user = self.client.get_user(owner)
        assert isinstance(user, User)

    def test_get_user_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_username = "abcde" * 10
            self.client.get_user(wrong_username)


###########
# Dataset #
###########
@pytest.mark.usefixtures("setup_class_client")
class TestDataset:
    def test_get_datasets(self) -> None:
        datasets = self.client.get_datasets()
        for dataset in datasets:
            assert isinstance(dataset, Dataset)

    def test_get_dataset(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            dataset = self.client.get_dataset(dataset_identifier)
            assert isinstance(dataset, Dataset)

    def test_get_dataset_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            self.client.get_dataset(wrong_dataset_identifier)

    def test_add_update_delete_dataset(self, add_update_dataset_arguments, owner) -> None:
        arguments = add_update_dataset_arguments
        try:
            # Add dataset
            dataset = self.client.add_dataset(**arguments)
            assert isinstance(dataset, Dataset)
            # Update dataset
            arguments["dataset_identifier"] = f"{owner}/{arguments['name']}"
            del arguments["name"]
            del arguments["organization"]
            dataset = self.client.update_dataset(**arguments)
            assert isinstance(dataset, Dataset)
        except AlreadyExistsError:
            pass
        finally:
            # Delete dataset
            self.client.delete_dataset(f"{owner}/add_dataset")

    def test_update_dataset_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            self.client.update_dataset(wrong_dataset_identifier)

    def test_delete_dataset_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            self.client.delete_dataset(wrong_dataset_identifier)

    def test_clone_dataset_networkerror(self) -> None:
        with pytest.raises(NetworkError):
            wrong_dataset_identifier = "abcde"
            self.client.clone_dataset(wrong_dataset_identifier)

    def test_clone_dataset_defaults(self, owner) -> None:
        name = "example-images-segmentation"
        dataset_identifier = f"{owner}/{name}"
        try:
            clone = self.client.clone_dataset(dataset_identifier, organization=owner)

            assert isinstance(clone, Dataset)
            assert clone.name == "example-images-segmentation-clone"
            assert clone.task_type == "segmentation-bitmap"
            assert not clone.public
        except AlreadyExistsError:
            pass
        finally:
            # Delete dataset
            self.client.delete_dataset(f"{owner}/{name}-clone")

    def test_clone_dataset_custom(self, owner) -> None:
        dataset_identifier = f"{owner}/example-images-vector"

        new_name = "example-images-vector-clone"
        new_task_type = TaskType.VECTOR

        try:
            clone = self.client.clone_dataset(
                dataset_identifier,
                new_name=new_name,
                new_task_type=new_task_type,
                organization=owner,
            )

            assert isinstance(clone, Dataset)
            assert clone.name == new_name
            assert clone.task_type == new_task_type
        except AlreadyExistsError:
            pass
        finally:
            # Delete dataset
            self.client.delete_dataset(f"{owner}/{new_name}")

    def test_get_add_update_delete_dataset_collaborator(self, owner, datasets) -> None:
        username = "admin-arnaud"
        role = Role.ADMIN
        new_role = Role.REVIEWER
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            try:
                # Add collaborator
                collaborator = self.client.add_dataset_collaborator(dataset_identifier, username, role)
                assert isinstance(collaborator, Collaborator)
                # Get collaborator
                collaborator = self.client.get_dataset_collaborator(dataset_identifier, username)
                assert isinstance(collaborator, Collaborator)
                # Update collaborator
                collaborator = self.client.update_dataset_collaborator(dataset_identifier, username, new_role)
                assert isinstance(collaborator, Collaborator)
            except AlreadyExistsError:
                pass
            finally:
                # Delete collaborator
                self.client.delete_dataset_collaborator(dataset_identifier, username)

    def test_delete_dataset_collaborator_notfounderror(self, owner, datasets) -> None:
        # Wrong dataset identifier and wrong username
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            wrong_username = "abcde"
            self.client.delete_dataset_collaborator(wrong_dataset_identifier, wrong_username)
        # Right dataset identifier and wrong username
        with pytest.raises(NotFoundError):
            right_dataset_identifier = f"{owner}/{datasets[0]}"
            wrong_username = "abcde"
            self.client.delete_dataset_collaborator(right_dataset_identifier, wrong_username)


##########
# Sample #
##########
@pytest.mark.usefixtures("setup_class_client")
class TestSample:
    def test_get_samples(self, datasets, owner) -> None:
        name = None
        label_status = LabelStatus.UNLABELED
        metadata = None
        sort: Final = "created"
        direction: Final = "desc"
        labelset = None  # "ground-truth"
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            samples = self.client.get_samples(
                dataset_identifier, labelset, name, label_status, metadata, sort, direction
            )
            for sample in samples:
                assert isinstance(sample, Sample)

    def test_get_samples_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            self.client.get_samples(wrong_dataset_identifier)

    def test_get_sample(self, sample_uuids) -> None:
        labelset = "ground-truth"
        for sample_uuid in sample_uuids:
            sample = self.client.get_sample(sample_uuid, labelset)
            assert isinstance(sample, Sample)

    def test_get_sample_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_uuid = "12345"
            self.client.get_sample(wrong_uuid)

    def test_add_update_delete_sample(
        self, datasets, owner, sample_attribute_types, TIME_INTERVAL, add_update_sample_arguments
    ) -> None:
        attributes_dict = add_update_sample_arguments

        metadata = {"weather": "sunny", "camera_id": 3}
        priority = 0
        name = "Test sample"

        # check if test sample already exists, if so delete it
        for dataset in datasets:
            samples = self.client.get_samples(f"{owner}/{dataset}")
            if any(sample.name == "Test sample" for sample in samples):
                sample = next(sample for sample in samples if sample.name == "Test sample")
                self.client.delete_sample(sample.uuid)

        for sample_attribute_type, dataset in zip(sample_attribute_types, datasets):
            dataset_identifier = f"{owner}/{dataset}"
            attributes = attributes_dict[sample_attribute_type]
            try:
                sample = self.client.add_sample(
                    dataset_identifier,
                    name,
                    attributes,
                    metadata,
                    priority,
                )
                assert isinstance(sample, Sample)
                sample = self.client.update_sample(
                    sample.uuid,
                    name,
                    attributes,
                    metadata,
                    priority,
                )
                assert isinstance(sample, Sample)
            finally:
                time.sleep(TIME_INTERVAL)
                self.client.delete_sample(sample.uuid)

            # Bulk endpoint
            returned_samples = None
            try:
                bulk_samples = [
                    {"name": f"sample_{1}", "attributes": attributes},
                    {"name": f"sample_{2}", "attributes": attributes},
                ]
                returned_samples = self.client.add_samples(dataset_identifier, bulk_samples)
            except AlreadyExistsError:
                pass
            finally:
                if returned_samples:
                    for sample in returned_samples:
                        self.client.delete_sample(sample.uuid)

    def test_update_sample_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_uuid = "12345"
            self.client.update_sample(wrong_uuid)

    def test_delete_sample_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_uuid = "12345"
            self.client.delete_sample(wrong_uuid)

    def test_unset_assigned(self, datasets, owner, add_update_sample_arguments, sample_attribute_types) -> None:
        attributes = add_update_sample_arguments
        name = "test_unset_assigned"

        attributes_type = sample_attribute_types[0]
        dataset = datasets[0]
        dataset_identifier = f"{owner}/{dataset}"

        if matched_samples := self.client.get_samples(dataset_identifier, name=name):
            assert len(matched_samples) == 1, f"Multiple matches found for sample with name {name}"
            self.client.delete_sample(matched_samples[0].uuid)

        sample = self.client.add_sample(dataset_identifier, name, attributes[attributes_type], assigned_reviewer=owner, assigned_labeler=owner)
        try:
            assert sample.assigned_labeler == owner
            assert sample.assigned_reviewer == owner

            sample = self.client.update_sample(sample.uuid, assigned_labeler=None)

            assert sample.assigned_labeler is None
            assert sample.assigned_reviewer == owner
            
            sample = self.client.update_sample(sample.uuid, assigned_reviewer=None)

            assert sample.assigned_reviewer is None
        finally:
            self.client.delete_sample(sample.uuid)


#########
# Label #
#########
@pytest.mark.usefixtures("setup_class_client")
class TestLabel:
    def test_add_update_get_delete_label(
        self, sample_uuids, label_attribute_types, label_test_attributes, TIME_INTERVAL
    ) -> None:
        labelset = "ground-truth"
        label_status = LabelStatus.PRELABELED
        score = 1
        for sample_uuid, label_attribute_type in zip(sample_uuids, label_attribute_types):
            attributes = label_test_attributes[label_attribute_type]
            try:
                # Add
                label = self.client.add_label(sample_uuid, labelset, attributes, label_status, score)
                assert isinstance(label, Label)
                # Update
                label = self.client.update_label(sample_uuid, labelset, attributes, label_status, score)
                assert isinstance(label, Label)
                # Get
                label = self.client.get_label(sample_uuid, labelset)
                assert isinstance(label, Label)
            except AlreadyExistsError:
                pass
            finally:
                # Delete
                time.sleep(TIME_INTERVAL)
                self.client.delete_label(sample_uuid, labelset)


#########
# Issue #
#########
@pytest.mark.usefixtures("setup_class_client")
class TestIssue:
    def test_get_issues(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            issues = self.client.get_issues(dataset_identifier)
            for issue in issues:
                assert isinstance(issue, Issue)

    def test_add_update_delete_issue(self, sample_uuids) -> None:
        description = "You forgot to label this car."
        for sample_uuid in sample_uuids:
            issue = None
            try:
                # Add issue
                issue = self.client.add_issue(sample_uuid, description)
                assert isinstance(issue, Issue)
                issue = self.client.update_issue(issue.uuid, description)
                assert isinstance(issue, Issue)
            except AlreadyExistsError:
                pass
            finally:
                # Delete issue
                if issue:
                    self.client.delete_issue(issue.uuid)

    def test_add_issue_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_sample_uuid = "12345"
            description = "You forgot to label this car."
            self.client.add_issue(wrong_sample_uuid, description)

    def test_update_issue_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_sample_uuid = "12345"
            description = "You forgot to label this car."
            self.client.update_issue(wrong_sample_uuid, description)


############
# Labelset #
############
@pytest.mark.usefixtures("setup_class_client")
class TestLabelset:
    def test_get_labelsets(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            labelsets = self.client.get_labelsets(dataset_identifier)
            for labelset in labelsets:
                assert isinstance(labelset, Labelset)

    def test_get_labelsets_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            self.client.get_labelsets(wrong_dataset_identifier)

    def test_get_labelset(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            labelsets = self.client.get_labelsets(dataset_identifier)
            for labelset in labelsets:
                labelset = self.client.get_labelset(dataset_identifier, labelset.name)
                assert isinstance(labelset, Labelset)

    def test_get_labelset_notfounderror(self, datasets, owner) -> None:
        # Wrong dataset identifier and wrong name
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.get_labelset(wrong_dataset_identifier, wrong_name)
        # Right dataset identifier and wrong name
        with pytest.raises(NotFoundError):
            right_dataset_identifier = f"{owner}/{datasets[0]}"
            wrong_name = "abcde"
            self.client.get_labelset(right_dataset_identifier, wrong_name)

    def test_add_delete_labelset(self, datasets, owner) -> None:
        name = "labelset4"
        description = "Test add_delete_labelset description."
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            try:
                # Add labelset
                labelset = self.client.add_labelset(dataset_identifier, name, description)
                assert isinstance(labelset, Labelset)
            except AlreadyExistsError:
                pass
            finally:
                # Delete labelset
                self.client.delete_labelset(dataset_identifier, name)

    def test_add_labelset_networkerror(self) -> None:
        with pytest.raises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.add_labelset(wrong_dataset_identifier, wrong_name)

    def test_delete_labelset_notfounderror(self, datasets, owner) -> None:
        # Wrong dataset identifier and wrong name
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.delete_labelset(wrong_dataset_identifier, wrong_name)
        # Right dataset identifier and wrong name
        with pytest.raises(NotFoundError):
            right_dataset_identifier = f"{owner}/{datasets[0]}"
            wrong_name = "abcde"
            self.client.delete_labelset(right_dataset_identifier, wrong_name)


###########
# Release #
###########
@pytest.mark.usefixtures("setup_class_client")
class TestRelease:
    def test_get_releases(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            releases = self.client.get_releases(dataset_identifier)
            for release in releases:
                assert isinstance(release, Release)

    def test_get_releases_notfounderror(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            self.client.get_releases(wrong_dataset_identifier)

    def test_get_release(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            releases = self.client.get_releases(dataset_identifier)
            for release in releases:
                release = self.client.get_release(dataset_identifier, release.name)
                assert isinstance(release, Release)

    def test_get_release_notfounderror(self, datasets, owner) -> None:
        # Wrong dataset identifier and wrong name
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.get_release(wrong_dataset_identifier, wrong_name)
        # Right dataset identifier and wrong name
        with pytest.raises(NotFoundError):
            right_dataset_identifier = f"{owner}/{datasets[0]}"
            wrong_name = "abcde"
            self.client.get_release(right_dataset_identifier, wrong_name)

    # @unittest.skip("No 3D segmentation release available")
    def test_add_delete_release(self, datasets, owner, TIME_INTERVAL) -> None:
        name = "v0.4"
        description = "Test release description."
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            dataset_info = self.client.get_dataset(dataset_identifier)
            if "pointcloud-segmentation" in dataset_info.task_type:
                # Release files not supported for 3D segmentation
                continue

            try:
                # Add release
                release = self.client.add_release(dataset_identifier, name, description)
                assert isinstance(release, Release)
            except AlreadyExistsError:
                pass
            finally:
                # Delete release
                time.sleep(TIME_INTERVAL)
                self.client.delete_release(dataset_identifier, name)

    def test_add_release_networkerror(self) -> None:
        # Wrong dataset identifier and wrong name
        with pytest.raises(NetworkError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.add_release(wrong_dataset_identifier, wrong_name)

    def test_delete_release_notfounderror(self, datasets, owner) -> None:
        # Wrong dataset identifier and wrong name
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            wrong_name = "abcde"
            self.client.delete_release(wrong_dataset_identifier, wrong_name)
        # Right dataset identifier and wrong name
        with pytest.raises(NotFoundError):
            right_dataset_identifier = f"{owner}/{datasets[0]}"
            wrong_name = "abcde"
            self.client.delete_release(right_dataset_identifier, wrong_name)


#########
# Asset #
#########
@pytest.mark.usefixtures("setup_class_client")
class TestAsset:
    def test_upload_asset(self) -> None:
        with open("tests/fixtures/test.png", "rb") as f:
            test_file = self.client.upload_asset(f, "test.png")
            assert isinstance(test_file, File)


##############
# Exceptions #
##############
@pytest.mark.usefixtures("setup_class_client")
class TestException:
    def test_wrong_api_key(self) -> None:
        with pytest.raises(AuthenticationError):
            wrong_api_key = "12345"
            SegmentsClient(wrong_api_key)

    def test_unauthorized_dataset_request(self, owner) -> None:
        with pytest.raises(AuthorizationError):
            dataset_identifier = f"{owner}/empty-dataset-for-authorization-tests"
            self.client.get_dataset(dataset_identifier)

    def test_dataset_already_exists_error(self, datasets, owner) -> None:
        organization = owner
        for dataset in datasets:
            with pytest.raises(AlreadyExistsError):
                self.client.add_dataset(dataset, organization=organization)

    def test_resource_not_found_error(self) -> None:
        with pytest.raises(NotFoundError):
            wrong_dataset_identifier = "abcde"
            self.client.get_dataset(wrong_dataset_identifier)

    def test_add_label_validationerror(self, sample_uuids) -> None:
        labelset = "ground-truth"
        wrong_attributes: Dict[str, Any] = {"wrong_key": "abcde"}
        for sample_uuid in sample_uuids:
            with pytest.raises(ValidationError):
                self.client.add_label(sample_uuid, labelset, wrong_attributes)


if __name__ == "__main__":
    unittest.main()
