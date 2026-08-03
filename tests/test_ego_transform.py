import math

import pytest
from segments.typing import (
    PCD,
    XYZ,
    XYZW,
    EgoPose,
    Label,
    LabelStatus,
    PCDType,
    PointcloudCuboidAnnotation,
    PointcloudCuboidLabelAttributes,
    PointcloudSampleAttributes,
    PointcloudSequenceCuboidAnnotation,
    PointcloudSequenceCuboidFrame,
    PointcloudSequenceCuboidLabelAttributes,
    PointcloudSequenceSampleAttributes,
    PointcloudVectorAnnotation,
    PointcloudVectorLabelAttributes,
    Sample,
    TaskType,
)
from segments.utils import transform_label_to_ego_coordinates


SAMPLE_UUID = "602a3eec-a61c-4a77-9fcc-3037ce5e9606"

IDENTITY_HEADING = XYZW(qx=0, qy=0, qz=0, qw=1)
# 90 degrees around the z-axis
YAW_90_HEADING = XYZW(qx=0, qy=0, qz=math.sin(math.pi / 4), qw=math.cos(math.pi / 4))


def make_sample(attributes) -> Sample:
    return Sample(
        uuid=SAMPLE_UUID,
        name="sample",
        attributes=attributes,
        metadata={},
        created_at="2024-01-01",
        created_by="tester",
        priority=0,
    )


def make_label(attributes, label_type: TaskType) -> Label:
    return Label(
        sample_uuid=SAMPLE_UUID,
        label_type=label_type,
        label_status=LabelStatus.LABELED,
        labelset="ground-truth",
        attributes=attributes,
        created_at="2024-01-01",
        created_by="tester",
        updated_at="2024-01-01",
    )


def make_pointcloud_attributes(ego_pose) -> PointcloudSampleAttributes:
    return PointcloudSampleAttributes(pcd=PCD(url="https://example.com/a.pcd", type=PCDType.PCD), ego_pose=ego_pose)


def make_cuboid(position: XYZ, yaw: float = 0.0, rotation=None) -> PointcloudCuboidAnnotation:
    return PointcloudCuboidAnnotation(
        id=1,
        category_id=1,
        position=position,
        dimensions=XYZ(x=1, y=1, z=1),
        yaw=yaw,
        rotation=rotation,
        type="cuboid",
    )


def assert_xyz_almost_equal(position: XYZ, expected) -> None:
    assert (position.x, position.y, position.z) == pytest.approx(expected)


def test_translation_only() -> None:
    sample = make_sample(make_pointcloud_attributes(EgoPose(position=XYZ(x=1, y=2, z=3), heading=IDENTITY_HEADING)))
    label = make_label(
        PointcloudCuboidLabelAttributes(annotations=[make_cuboid(XYZ(x=2, y=2, z=3))]),
        TaskType.POINTCLOUD_CUBOID,
    )

    transformed = transform_label_to_ego_coordinates(sample, label)

    assert_xyz_almost_equal(transformed.attributes.annotations[0].position, (1, 0, 0))
    # the original label is untouched
    assert_xyz_almost_equal(label.attributes.annotations[0].position, (2, 2, 3))


def test_rotation_and_yaw() -> None:
    sample = make_sample(make_pointcloud_attributes(EgoPose(position=XYZ(x=0, y=0, z=0), heading=YAW_90_HEADING)))
    label = make_label(
        PointcloudCuboidLabelAttributes(annotations=[make_cuboid(XYZ(x=1, y=0, z=0), yaw=0.0, rotation=IDENTITY_HEADING)]),
        TaskType.POINTCLOUD_CUBOID,
    )

    transformed = transform_label_to_ego_coordinates(sample, label)
    annotation = transformed.attributes.annotations[0]

    assert_xyz_almost_equal(annotation.position, (0, -1, 0))
    assert annotation.yaw == pytest.approx(-math.pi / 2)
    assert (annotation.rotation.qx, annotation.rotation.qy) == pytest.approx((0, 0))
    assert annotation.rotation.qz == pytest.approx(-math.sin(math.pi / 4))
    assert annotation.rotation.qw == pytest.approx(math.cos(math.pi / 4))


def test_vector_points() -> None:
    sample = make_sample(make_pointcloud_attributes(EgoPose(position=XYZ(x=1, y=0, z=0), heading=YAW_90_HEADING)))
    label = make_label(
        PointcloudVectorLabelAttributes(
            annotations=[
                PointcloudVectorAnnotation(
                    id=1, category_id=1, points=[[1, 0, 0], [2, 0, 0]], type="polyline"
                )
            ]
        ),
        TaskType.POINTCLOUD_VECTOR,
    )

    transformed = transform_label_to_ego_coordinates(sample, label)

    points = transformed.attributes.annotations[0].points
    assert points[0] == pytest.approx((0, 0, 0))
    assert points[1] == pytest.approx((0, -1, 0))


def test_sequence_with_missing_ego_pose(caplog) -> None:
    sample = make_sample(
        PointcloudSequenceSampleAttributes(
            frames=[
                make_pointcloud_attributes(EgoPose(position=XYZ(x=1, y=0, z=0), heading=IDENTITY_HEADING)),
                make_pointcloud_attributes(None),
            ]
        )
    )

    def make_frame(position: XYZ) -> PointcloudSequenceCuboidFrame:
        return PointcloudSequenceCuboidFrame(
            annotations=[
                PointcloudSequenceCuboidAnnotation(
                    id=1,
                    category_id=1,
                    position=position,
                    dimensions=XYZ(x=1, y=1, z=1),
                    yaw=0.0,
                    type="cuboid",
                    track_id=1,
                )
            ]
        )

    label = make_label(
        PointcloudSequenceCuboidLabelAttributes(frames=[make_frame(XYZ(x=2, y=0, z=0)), make_frame(XYZ(x=3, y=0, z=0))]),
        TaskType.POINTCLOUD_CUBOID_SEQUENCE,
    )

    transformed = transform_label_to_ego_coordinates(sample, label)

    assert_xyz_almost_equal(transformed.attributes.frames[0].annotations[0].position, (1, 0, 0))
    # frame without an ego pose is left in world coordinates
    assert_xyz_almost_equal(transformed.attributes.frames[1].annotations[0].position, (3, 0, 0))
    assert "no ego pose" in caplog.text


def test_uuid_mismatch_raises() -> None:
    sample = make_sample(make_pointcloud_attributes(None))
    label = make_label(
        PointcloudCuboidLabelAttributes(annotations=[make_cuboid(XYZ(x=0, y=0, z=0))]),
        TaskType.POINTCLOUD_CUBOID,
    )
    label.sample_uuid = "other-uuid"

    with pytest.raises(ValueError, match="does not match"):
        transform_label_to_ego_coordinates(sample, label)


def test_multisensor() -> None:
    from segments.typing import (
        BasePointcloudSequenceCuboidLabelAttributes,
        MultiSensorLabelAttributes,
        MultiSensorPointcloudSequenceCuboidLabelAttributes,
        MultiSensorPointcloudSequenceSampleAttributes,
        MultiSensorSampleAttributes,
    )

    sample = make_sample(
        MultiSensorSampleAttributes(
            sensors=[
                MultiSensorPointcloudSequenceSampleAttributes(
                    name="lidar",
                    task_type=TaskType.POINTCLOUD_CUBOID_SEQUENCE,
                    attributes=PointcloudSequenceSampleAttributes(
                        frames=[make_pointcloud_attributes(EgoPose(position=XYZ(x=1, y=0, z=0), heading=IDENTITY_HEADING))]
                    ),
                )
            ]
        )
    )
    label = make_label(
        MultiSensorLabelAttributes(
            sensors=[
                MultiSensorPointcloudSequenceCuboidLabelAttributes(
                    name="lidar",
                    task_type=TaskType.POINTCLOUD_CUBOID_SEQUENCE,
                    attributes=BasePointcloudSequenceCuboidLabelAttributes(
                        frames=[
                            PointcloudSequenceCuboidFrame(
                                annotations=[
                                    PointcloudSequenceCuboidAnnotation(
                                        id=1,
                                        category_id=1,
                                        position=XYZ(x=3, y=0, z=0),
                                        dimensions=XYZ(x=1, y=1, z=1),
                                        yaw=0.0,
                                        type="cuboid",
                                        track_id=1,
                                    )
                                ]
                            )
                        ]
                    ),
                )
            ]
        ),
        TaskType.MULTISENSOR_SEQUENCE,
    )

    transformed = transform_label_to_ego_coordinates(sample, label)

    annotation = transformed.attributes.sensors[0].attributes.frames[0].annotations[0]
    assert_xyz_almost_equal(annotation.position, (2, 0, 0))


def test_client_get_label_transform(monkeypatch) -> None:
    from segments.client import SegmentsClient

    sample = make_sample(make_pointcloud_attributes(EgoPose(position=XYZ(x=1, y=2, z=3), heading=IDENTITY_HEADING)))
    label = make_label(
        PointcloudCuboidLabelAttributes(annotations=[make_cuboid(XYZ(x=2, y=2, z=3))]),
        TaskType.POINTCLOUD_CUBOID,
    )

    # bypass __init__, which verifies the API key over the network
    client = SegmentsClient.__new__(SegmentsClient)
    monkeypatch.setattr(client, "_get", lambda *args, **kwargs: label)
    monkeypatch.setattr(client, "get_sample", lambda uuid: sample)

    untransformed = client.get_label(SAMPLE_UUID)
    assert_xyz_almost_equal(untransformed.attributes.annotations[0].position, (2, 2, 3))

    transformed = client.get_label(SAMPLE_UUID, transform_to_ego_coordinates=True)
    assert_xyz_almost_equal(transformed.attributes.annotations[0].position, (1, 0, 0))


def test_non_pointcloud_sample_raises() -> None:
    from segments.typing import ImageSampleAttributes, URL

    sample = make_sample(ImageSampleAttributes(image=URL(url="https://example.com/a.jpg")))
    label = make_label(
        PointcloudCuboidLabelAttributes(annotations=[make_cuboid(XYZ(x=0, y=0, z=0))]),
        TaskType.POINTCLOUD_CUBOID,
    )

    with pytest.raises(ValueError, match="point cloud"):
        transform_label_to_ego_coordinates(sample, label)
