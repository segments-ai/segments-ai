from typing import Any, Dict

import pydantic
import pytest
from pydantic import TypeAdapter
from segments.typing import LabelAttributes, Link, OrthoImage, SampleAttributes, TaskAttributes


def test_link_category_restrictions_round_trip() -> None:
    """The allowlist must survive the validate + dump path that add_dataset/update_dataset use."""
    task_attributes = TaskAttributes.model_validate(
        {
            "format_version": "0.1",
            "categories": [
                {"name": "car", "id": 1, "link_category_restrictions": [2]},
                {"name": "wheel", "id": 2, "link_category_restrictions": []},
                {"name": "tree", "id": 3},
            ],
        }
    )

    assert task_attributes.categories[0].link_category_restrictions == [2]
    assert task_attributes.categories[1].link_category_restrictions == []
    # Omitted stays omitted rather than becoming an empty allowlist, which would mean "link to nothing".
    assert task_attributes.categories[2].link_category_restrictions is None

    dumped = task_attributes.model_dump(mode="json", exclude_unset=True)
    assert dumped["categories"][0]["link_category_restrictions"] == [2]
    assert dumped["categories"][1]["link_category_restrictions"] == []
    assert "link_category_restrictions" not in dumped["categories"][2]


LINK_WITH_FRAMES: Dict[str, Any] = {
    "from_id": 3,
    "to_id": 7,
    "attributes": {"relation": "follows"},
    "frames": [
        {"attributes": {"distance": 12.5}},
        {"attributes": {"distance": 11.9}},
        {"attributes": {}},
    ],
}


def test_link_frames_round_trip() -> None:
    """Frame-level link attribute values must survive the validate + dump path that add_label/update_label use."""
    dumped = Link.model_validate(LINK_WITH_FRAMES).model_dump(mode="json", exclude_unset=True)

    assert dumped == LINK_WITH_FRAMES


def test_link_without_frames_dumps_without_frames_key() -> None:
    """A link that never had frame-level values must not start sending ``"frames": null``."""
    dumped = Link.model_validate({"from_id": 3, "to_id": 7, "attributes": {}}).model_dump(
        mode="json", exclude_unset=True
    )

    assert "frames" not in dumped


def test_label_attributes_with_link_frames_round_trip() -> None:
    """The exact TypeAdapter path used by add_label/update_label must keep frame-level link values."""
    label_attributes = {
        "format_version": "0.2",
        "frames": [
            {
                "annotations": [
                    {
                        "id": 1,
                        "category_id": 1,
                        "track_id": 3,
                        "type": "cuboid",
                        "position": {"x": 0, "y": 0, "z": 0},
                        "dimensions": {"x": 1, "y": 1, "z": 1},
                        "yaw": 0,
                    }
                ],
            }
            for _ in range(3)
        ],
        "links": [LINK_WITH_FRAMES],
    }

    dumped = TypeAdapter(LabelAttributes).validate_python(label_attributes).model_dump(mode="json", exclude_unset=True)

    assert dumped["links"] == [LINK_WITH_FRAMES]


def _task_attributes_with_link_attribute(link_attribute: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "format_version": "0.1",
        "categories": [
            {
                "name": "car",
                "id": 1,
                "link_attributes": [{"name": "distance", "input_type": "number", **link_attribute}],
            }
        ],
    }


def test_frame_level_link_attribute_definition_round_trip() -> None:
    """``is_track_level: false`` is how the web app stores frame-level link attributes; it must validate and be kept."""
    task_attributes = TaskAttributes.model_validate(_task_attributes_with_link_attribute({"is_track_level": False}))

    assert task_attributes.categories[0].link_attributes[0].is_track_level is False

    dumped = task_attributes.model_dump(mode="json", exclude_unset=True)
    assert dumped["categories"][0]["link_attributes"][0]["is_track_level"] is False


def test_link_attribute_definition_defaults_to_sequence_level() -> None:
    """A missing ``is_track_level`` means sequence-level, matching the web app and legacy labels."""
    task_attributes = TaskAttributes.model_validate(_task_attributes_with_link_attribute({}))

    assert task_attributes.categories[0].link_attributes[0].is_track_level is True


def test_link_attribute_definition_without_is_track_level_dumps_unchanged() -> None:
    """Omitted must stay omitted so an update_dataset round-trip does not rewrite existing definitions."""
    dumped = TaskAttributes.model_validate(_task_attributes_with_link_attribute({})).model_dump(
        mode="json", exclude_unset=True
    )

    assert "is_track_level" not in dumped["categories"][0]["link_attributes"][0]


@pytest.mark.parametrize("is_track_level", [True, None])
def test_link_attribute_definition_accepts_explicit_sequence_level(is_track_level: Any) -> None:
    task_attributes = TaskAttributes.model_validate(
        _task_attributes_with_link_attribute({"is_track_level": is_track_level})
    )

    assert task_attributes.categories[0].link_attributes[0].is_track_level is is_track_level


CORNERS: Dict[str, Any] = {
    "top_left": {"x": -10.0, "y": 10.0},
    "top_right": {"x": 10.0, "y": 10.0},
    "bottom_right": {"x": 10.0, "y": -10.0},
    "bottom_left": {"x": -10.0, "y": -10.0},
}
HEIGHT_MAP: Dict[str, Any] = {
    "url": "https://example.com/ortho_height.png",
    "encoding": "rgb16",
    "offset": -12.5,
    "scale": 0.01,
}
ORTHO_IMAGE: Dict[str, Any] = {"url": "https://example.com/ortho.png", "corners": CORNERS}


def dump_like_client(ortho_image: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and dump point cloud sample attributes the way ``SegmentsClient.add_sample`` does."""
    attributes = {"pcd": {"url": "url", "type": "kitti"}, "ortho_images": [ortho_image]}
    dumped = TypeAdapter(SampleAttributes).validate_python(attributes).model_dump(mode="json", exclude_unset=True)
    return dumped["ortho_images"][0]


def test_ortho_image_height_map_round_trip() -> None:
    ortho_image = {**ORTHO_IMAGE, "height_map": HEIGHT_MAP}

    assert dump_like_client(ortho_image) == ortho_image


def test_ortho_image_without_height_map_dumps_without_height_map_key() -> None:
    """Existing ortho images must not start sending ``"height_map": null``."""
    assert "height_map" not in dump_like_client(ORTHO_IMAGE)


def test_height_map_sends_default_offset_and_scale() -> None:
    """The server applies no defaults, so the client must send them even when they were not set."""
    height_map = {k: v for k, v in HEIGHT_MAP.items() if k not in ("offset", "scale")}

    dumped = dump_like_client({**ORTHO_IMAGE, "height_map": height_map})

    assert dumped["height_map"] == {**height_map, "offset": 0, "scale": 1}


@pytest.mark.parametrize(
    "height_map",
    [
        pytest.param({k: v for k, v in HEIGHT_MAP.items() if k != "encoding"}, id="missing encoding"),
        pytest.param({**HEIGHT_MAP, "encoding": "png16"}, id="unknown encoding"),
        pytest.param({**HEIGHT_MAP, "scale": 0}, id="zero scale"),
        pytest.param({**HEIGHT_MAP, "scale": -0.01}, id="negative scale"),
        pytest.param({**HEIGHT_MAP, "scale": float("inf")}, id="infinite scale"),
        pytest.param({**HEIGHT_MAP, "offset": float("nan")}, id="nan offset"),
    ],
)
def test_height_map_rejects_invalid(height_map: Dict[str, Any]) -> None:
    with pytest.raises(pydantic.ValidationError):
        OrthoImage.model_validate({**ORTHO_IMAGE, "height_map": height_map})
