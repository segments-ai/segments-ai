import time
import typing
from typing import Final

import pydantic
import pytest
import segments
import segments.typing as segments_typing
from segments.client import SegmentsClient
from segments.exceptions import AlreadyExistsError, CollaboratorError, InvalidModelError
from segments.resource_api import Collaborator, Dataset, Issue, Label, Labelset, Release, Sample
from segments.typing import (
    LabelStatus,
    PointcloudSegmentationLabelAttributes,
    Role,
    TaskType,
)


@pytest.mark.parametrize(
    "func1,func2,ignore",
    [
        (Dataset.add_sample, SegmentsClient.add_sample, ["dataset_identifier"]),
        (Dataset.delete, SegmentsClient.delete_dataset, ["dataset_identifier"]),
        (Dataset.update, SegmentsClient.update_dataset, ["dataset_identifier"]),
        (Dataset.get_samples, SegmentsClient.get_samples, ["dataset_identifier"]),
        (Dataset.get_collaborator, SegmentsClient.get_dataset_collaborator, ["dataset_identifier"]),
        (Dataset.add_collaborator, SegmentsClient.add_dataset_collaborator, ["dataset_identifier"]),
        (Dataset.add_release, SegmentsClient.add_release, ["dataset_identifier"]),
        (Dataset.get_release, SegmentsClient.get_release, ["dataset_identifier"]),
        (Dataset.get_releases, SegmentsClient.get_releases, ["dataset_identifier"]),
        (Dataset.get_labelsets, SegmentsClient.get_labelsets, ["dataset_identifier"]),
        (Dataset.get_labelset, SegmentsClient.get_labelset, ["dataset_identifier"]),
        (Dataset.add_labelset, SegmentsClient.add_labelset, ["dataset_identifier"]),
        (Dataset.get_issues, SegmentsClient.get_issues, ["dataset_identifier"]),
        (Dataset.get_workunits, SegmentsClient.get_workunits, ["dataset_identifier"]),
        (Dataset.clone, SegmentsClient.clone_dataset, ["dataset_identifier"]),
        (Sample.delete, SegmentsClient.delete_sample, ["uuid"]),
        (Sample.update, SegmentsClient.update_sample, ["uuid"]),
        (Sample.get_label, SegmentsClient.get_label, ["sample_uuid"]),
        (Sample.add_label, SegmentsClient.add_label, ["sample_uuid"]),
        (Sample.update_label, SegmentsClient.update_label, ["sample_uuid"]),
        (Sample.delete_label, SegmentsClient.delete_label, ["sample_uuid"]),
        (Sample.add_issue, SegmentsClient.add_issue, ["sample_uuid"]),
        (Label.update, SegmentsClient.update_label, ["sample_uuid", "labelset"]),
        (Label.delete, SegmentsClient.delete_label, ["sample_uuid", "labelset"]),
        (Issue.delete, SegmentsClient.delete_issue, ["uuid"]),
        (Issue.update, SegmentsClient.update_issue, ["uuid"]),
        (Collaborator.update, SegmentsClient.update_dataset_collaborator, ["dataset_identifier", "username"]),
        (Collaborator.delete, SegmentsClient.delete_dataset_collaborator, ["dataset_identifier", "username"]),
        (Labelset.delete, SegmentsClient.delete_labelset, ["dataset_identifier", "name"]),
        (Release.delete, SegmentsClient.delete_release, ["dataset_identifier", "name"]),
    ],
)
def test_same_arguments(func1, func2, ignore: list[str]):
    sig1 = typing.get_type_hints(func1)
    sig2 = typing.get_type_hints(func2)

    ignored_arguments_set = set(ignore)
    func1_argnames = sig1.keys() - ignored_arguments_set
    func2_argnames = sig2.keys() - ignored_arguments_set
    if func1_argnames != func2_argnames:
        raise AssertionError(
            f"Arguments in {func1.__name__} and {func2.__name__} are not the same: {func1_argnames ^ func2_argnames}"
        )


@pytest.mark.usefixtures("setup_class_client")
class TestDataset:
    def test_add_update_delete_dataset(self, add_update_dataset_arguments) -> None:
        arguments = add_update_dataset_arguments
        dataset = None
        try:
            # Add dataset
            dataset = self.client.add_dataset(**arguments)
            assert isinstance(dataset, Dataset)
            # Update dataset
            del arguments["name"]
            del arguments["organization"]
            dataset = dataset.update(**arguments)
            assert isinstance(dataset, Dataset)
        except AlreadyExistsError:
            pass
        finally:
            # Delete dataset
            if dataset is not None:
                dataset.delete()

    def test_clone_dataset_defaults(self, owner) -> None:
        name = "example-images-segmentation"
        dataset_identifier = f"{owner}/{name}"
        clone = None
        try:
            dataset = self.client.get_dataset(dataset_identifier)
            clone = dataset.clone()

            assert isinstance(clone, Dataset)
            assert clone.name == "example-images-segmentation-clone"
            assert clone.task_type == "segmentation-bitmap"
            assert not clone.public
        except AlreadyExistsError:
            pass
        finally:
            # Delete dataset
            if clone is not None:
                clone.delete()

    def test_clone_dataset_custom(self, owner) -> None:
        dataset_identifier = f"{owner}/example-images-vector"

        new_name = "example-images-vector-clone"
        new_task_type = TaskType.VECTOR
        clone = None

        try:
            dataset = self.client.get_dataset(dataset_identifier)
            clone = dataset.clone(
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
            if clone is not None:
                clone.delete()

    def test_get_add_update_delete_dataset_collaborator(self, owner, datasets) -> None:
        username = "admin-arnaud"
        role = Role.ADMIN
        dataset_name = datasets[0]
        dataset_identifier = f"{owner}/{dataset_name}"

        dataset = self.client.get_dataset(dataset_identifier)
        collaborator = None
        try:
            # Add collaborator
            collaborator = dataset.add_collaborator(username, role)
            assert isinstance(collaborator, segments_typing.Collaborator)
            # Get collaborator
            collaborator_get = dataset.get_collaborator(username)
            assert isinstance(collaborator, segments_typing.Collaborator)
            assert collaborator == collaborator_get

            collaborator_update = collaborator.update(Role.ADMIN)
            assert collaborator_update.role == Role.ADMIN
        except CollaboratorError as e:
            # Collaborator already exists, clean it up and fail the test. This should let the test pass on the next run.
            self.client.delete_dataset_collaborator(dataset_identifier, username)
            raise e
        finally:
            # Delete collaborator
            if collaborator is not None:
                collaborator.delete()


@pytest.mark.usefixtures("setup_class_client")
class TestSample:
    def test_get_samples(self, datasets, owner) -> None:
        name = None
        label_status = LabelStatus.UNLABELED
        metadata = None
        sort: Final = "created"
        direction: Final = "desc"
        labelset = None  # "ground-truth"
        for dataset_name in datasets:
            dataset_identifier = f"{owner}/{dataset_name}"
            dataset = self.client.get_dataset(dataset_identifier)
            samples = dataset.get_samples(dataset_identifier, labelset, name, label_status, metadata, sort, direction)
            for sample in samples:
                assert isinstance(sample, Sample)

    def test_add_update_delete_sample(
        self, datasets, owner, sample_attribute_types, TIME_INTERVAL, add_update_sample_arguments
    ) -> None:
        attributes_dict = add_update_sample_arguments

        metadata = {"weather": "sunny", "camera_id": 3}
        priority = 0
        name = "Test sample"

        # check if test sample already exists, if so delete it
        for dataset_name in datasets:
            dataset = self.client.get_dataset(f"{owner}/{dataset_name}")
            samples = dataset.get_samples()
            if any(sample.name == "Test sample" for sample in samples):
                sample = next(sample for sample in samples if sample.name == "Test sample")
                sample.delete()

        for sample_attribute_type, dataset_name in zip(sample_attribute_types, datasets):
            dataset_identifier = f"{owner}/{dataset_name}"
            dataset = self.client.get_dataset(dataset_identifier)
            attributes = attributes_dict[sample_attribute_type]
            sample = None
            try:
                sample = dataset.add_sample(
                    name,
                    attributes,
                    metadata,
                    priority,
                )
                assert isinstance(sample, Sample)
                sample = sample.update(
                    name,
                    attributes,
                    metadata,
                    priority,
                )
                assert isinstance(sample, Sample)
            finally:
                if sample is not None:
                    time.sleep(TIME_INTERVAL)
                    sample.delete()

    def test_unset_assigned(self, datasets, owner, add_update_sample_arguments, sample_attribute_types) -> None:
        attributes = add_update_sample_arguments
        name = "test_unset_assigned"

        attributes_type = sample_attribute_types[0]
        dataset_name = datasets[0]
        dataset_identifier = f"{owner}/{dataset_name}"

        dataset = self.client.get_dataset(dataset_identifier)

        if matched_samples := dataset.get_samples(name=name):
            assert len(matched_samples) == 1, f"Multiple matches found for sample with name {name}"
            matched_samples[0].delete()

        sample = dataset.add_sample(name, attributes[attributes_type], assigned_reviewer=owner, assigned_labeler=owner)
        try:
            assert sample.assigned_labeler == owner
            assert sample.assigned_reviewer == owner

            sample = sample.update(assigned_labeler=None)

            assert sample.assigned_labeler is None
            assert sample.assigned_reviewer == owner

            sample = sample.update(assigned_reviewer=None)

            assert sample.assigned_reviewer is None
        finally:
            sample.delete()


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
            sample = self.client.get_sample(sample_uuid)
            label = None
            try:
                # Add
                label = sample.add_label(labelset, attributes, label_status, score)
                assert isinstance(label, Label)
                # Update
                label = label.update(attributes, label_status, score)
                assert isinstance(label, Label)
                # Also test the sample.update_label interface
                label = sample.update_label(labelset, attributes, label_status, score)
                assert isinstance(label, Label)
                label_get = sample.get_label(labelset)
                assert label.attributes == label_get.attributes
            except AlreadyExistsError:
                pass
            finally:
                if label is not None:
                    # Delete
                    time.sleep(TIME_INTERVAL)
                    label.delete()

    def test_validation_error(self, sample_uuids, label_attribute_types, bad_image_segmentation_label, TIME_INTERVAL):
        sample_idx = label_attribute_types.index("image-segmentation")
        sample_uuid = sample_uuids[sample_idx]
        sample = self.client.get_sample(sample_uuid)
        label = None
        with pytest.raises(pydantic.ValidationError) as excinfo:
            try:
                label = sample.add_label("ground-truth", bad_image_segmentation_label)
            finally:
                if label is not None:
                    time.sleep(TIME_INTERVAL)
                    label.delete()

        # Convenience api should only throw one error
        assert excinfo.value.error_count() == 1

        with pytest.raises(segments.exceptions.ValidationError) as excinfo:
            try:
                label = self.client.add_label(sample.uuid, "ground-truth", bad_image_segmentation_label)
            finally:
                if label is not None:
                    time.sleep(TIME_INTERVAL)
                    label.delete()

        # We expect the client api to throw many more errors
        assert excinfo.value.cause.error_count() > 1

    def test_provide_wrong_model(self, sample_uuids, label_attribute_types, label_test_attributes):
        # We'll try to provide a pointcloud segmentation model for an image segmentation label
        sample_idx = label_attribute_types.index("image-segmentation")
        sample_uuid = sample_uuids[sample_idx]

        bad_attributes = label_test_attributes["pointcloud-segmentation"]

        sample = self.client.get_sample(sample_uuid)
        with pytest.raises(InvalidModelError):
            sample.add_label("ground-truth", PointcloudSegmentationLabelAttributes.model_validate(bad_attributes))

    def test_vector_label_attributes_contains_track_id(
        self, sample_uuids, label_attribute_types, label_test_attributes
    ):
        task_type = "pointcloud-sequence-vector"
        sample_idx = label_attribute_types.index(task_type)
        sample_uuid = sample_uuids[sample_idx]
        sample = self.client.get_sample(sample_uuid)

        try:
            sample.add_label(labelset="ground-truth", attributes=label_test_attributes[task_type])
            label = sample.get_label()

            for annotation in label.attributes.frames[0].annotations:
                assert isinstance(annotation.track_id, int)
        finally:
            sample.delete_label(labelset="ground-truth")


@pytest.mark.usefixtures("setup_class_client")
class TestIssue:
    def test_get_issues(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            dataset = self.client.get_dataset(dataset_identifier)
            issues = dataset.get_issues()
            for issue in issues:
                assert isinstance(issue, segments_typing.Issue)

    def test_add_update_delete_issue(self, sample_uuids) -> None:
        description = "You forgot to label this car."
        sample_uuid = sample_uuids[0]
        sample = self.client.get_sample(sample_uuid)
        issue = None
        try:
            # Add issue
            issue = sample.add_issue(description)
            assert isinstance(issue, segments_typing.Issue)
            description2 = description.replace("car", "bike")
            issue = issue.update(description2)
            assert isinstance(issue, segments_typing.Issue)
            assert issue.description == description2
        finally:
            # Delete issue
            if issue:
                issue.delete()


@pytest.mark.usefixtures("setup_class_client")
class TestRelease:
    def test_get_releases(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            dataset = self.client.get_dataset(dataset_identifier)
            releases = dataset.get_releases()
            for release in releases:
                assert isinstance(release, Release)

    def test_get_release(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            dataset = self.client.get_dataset(dataset_identifier)
            releases = dataset.get_releases()
            for release in releases:
                release = dataset.get_release(release.name)
                assert isinstance(release, Release)

    def test_add_delete_release(self, datasets, owner, TIME_INTERVAL) -> None:
        name = "v0.4"
        description = "Test release description."
        for dataset_name in datasets:
            dataset_identifier = f"{owner}/{dataset_name}"
            dataset = self.client.get_dataset(dataset_identifier)
            if "pointcloud-segmentation" in dataset.task_type:
                # Release files not supported for 3D segmentation
                continue

            release = None
            try:
                # Add release
                release = dataset.add_release(name, description)
                assert isinstance(release, Release)
            except AlreadyExistsError as e:
                release = dataset.get_release(name)
                raise e
            finally:
                # Delete release
                time.sleep(TIME_INTERVAL)
                if release is not None:
                    release.delete()


@pytest.mark.usefixtures("setup_class_client")
class TestLabelset:
    def test_get_labelsets(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            dataset = self.client.get_dataset(dataset_identifier)
            labelsets = dataset.get_labelsets()
            for labelset in labelsets:
                assert isinstance(labelset, Labelset)

    def test_get_labelset(self, datasets, owner) -> None:
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            dataset = self.client.get_dataset(dataset_identifier)
            labelsets = dataset.get_labelsets()
            for labelset in labelsets:
                labelset = dataset.get_labelset(labelset.name)
                assert isinstance(labelset, Labelset)

    def test_add_delete_labelset(self, datasets, owner) -> None:
        name = "labelset4"
        description = "Test add_delete_labelset description."
        for dataset in datasets:
            dataset_identifier = f"{owner}/{dataset}"
            dataset = self.client.get_dataset(dataset_identifier)
            labelset = None
            try:
                # Add labelset
                labelset = dataset.add_labelset(name, description)
                assert isinstance(labelset, Labelset)
            except AlreadyExistsError as e:
                labelset = dataset.get_labelset(name)
                raise e
            finally:
                # Delete labelset
                if labelset is not None:
                    labelset.delete()
