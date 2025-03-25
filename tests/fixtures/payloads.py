from typing import Any, Dict

from pytest import fixture


@fixture
def add_update_dataset_arguments(owner):
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
        "organization": owner,
    }
    return arguments


@fixture
def add_update_sample_arguments():
    attributes_dict = {
        "image": {"image": {"url": "url"}},
        "image-sequence": {"frames": [{"image": {"url": "url"}}, {"image": {"url": ""}}]},
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
    }
    # Multi-sensor
    attributes_dict["multi-sensor"] = {
        "sensors": [
            {
                "name": "Lidar",
                "task_type": "pointcloud-cuboid-sequence",
                "attributes": attributes_dict["pointcloud-sequence"],
            },
            {
                "name": "Camera 1",
                "task_type": "image-vector-sequence",
                "attributes": attributes_dict["image-sequence"],
            },
            {
                "name": "Camera 2",
                "task_type": "image-vector-sequence",
                "attributes": attributes_dict["image-sequence"],
            },
        ]
    }

    return attributes_dict


@fixture
def label_image_or_object_attributes():
    image_or_object_attributes = {  # sample-level attributes
        "scene_type": "crossroads",
        "weather": "sunny",
        "isRaining": True,
    }
    return image_or_object_attributes


@fixture
def label_test_attributes(label_image_or_object_attributes):
    label_attributes: Dict[str, Dict[str, Any]] = {
        "image-segmentation": {
            "format_version": "0.1",
            "annotations": [
                {
                    "id": 1,
                    "category_id": 1,
                    "attributes": label_image_or_object_attributes,
                },
                {
                    "id": 2,
                    "category_id": 1,
                    "attributes": label_image_or_object_attributes,
                },
                {
                    "id": 3,
                    "category_id": 4,
                    "attributes": label_image_or_object_attributes,
                },
            ],
            "segmentation_bitmap": {
                "url": "https://segmentsai-staging.s3.eu-west-2.amazonaws.com/assets/davy/ddf55e99-1a6f-42d2-83e9-8657de3259a1.png"
            },
            "image_attributes": label_image_or_object_attributes,
        },
        "image-vector": {
            "format_version": "0.1",
            "annotations": [
                {
                    "id": 1,
                    "category_id": 1,
                    "type": "bbox",
                    "points": [[12.34, 56.78], [90.12, 34.56]],
                    "attributes": label_image_or_object_attributes,
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
                    "attributes": label_image_or_object_attributes,
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
                    "attributes": label_image_or_object_attributes,
                },
                {
                    "id": 4,
                    "category_id": 4,
                    "type": "point",
                    "points": [[12.34, 56.78]],
                    "attributes": label_image_or_object_attributes,
                },
            ],
            "image_attributes": label_image_or_object_attributes,
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
                            "attributes": label_image_or_object_attributes,
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
                            "attributes": label_image_or_object_attributes,
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
                            "attributes": label_image_or_object_attributes,
                        },
                        {
                            "id": 4,
                            "category_id": 4,
                            "track_id": 3,
                            "is_keyframe": True,
                            "type": "point",
                            "points": [[12.34, 56.78]],
                            "attributes": label_image_or_object_attributes,
                        },
                    ],
                    "image_attributes": label_image_or_object_attributes,
                }
            ],
        },
        "pointcloud-segmentation": {
            "format_version": "0.1",
            "annotations": [
                {
                    "id": 1,
                    "category_id": 1,
                    "attributes": label_image_or_object_attributes,
                },
                {
                    "id": 2,
                    "category_id": 1,
                    "attributes": label_image_or_object_attributes,
                },
                {
                    "id": 3,
                    "category_id": 4,
                    "attributes": label_image_or_object_attributes,
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
    }
    # Multi-sensor
    label_attributes["multi-sensor"] = {
        "sensors": [
            {
                "name": "Lidar",
                "task_type": "pointcloud-cuboid-sequence",
                "attributes": label_attributes["pointcloud-sequence-cuboid"],
            },
            {
                "name": "Camera 1",
                "task_type": "image-vector-sequence",
                "attributes": label_attributes["image-sequence-vector"],
            },
            {
                "name": "Camera 2",
                "task_type": "image-vector-sequence",
                "attributes": label_attributes["image-sequence-vector"],
            },
        ]
    }

    return label_attributes


@fixture
def bad_image_segmentation_label(label_image_or_object_attributes):
    attributes = {
        "format_version": "0.1",
        "annotations": [
            {
                "id": 1,
                "category_id": 1,
                "attributes": label_image_or_object_attributes,
            },
            {
                "id": 2,
                "category_id": None,
                "attributes": label_image_or_object_attributes,
            },
            {
                "id": 3,
                "category_id": 4,
                "attributes": label_image_or_object_attributes,
            },
        ],
        "segmentation_bitmap": {
            "url": "https://segmentsai-staging.s3.eu-west-2.amazonaws.com/assets/davy/ddf55e99-1a6f-42d2-83e9-8657de3259a1.png"
        },
        "image_attributes": label_image_or_object_attributes,
    }
    return attributes
