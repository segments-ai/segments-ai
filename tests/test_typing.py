from typing import Any, Dict

import pytest
from pydantic import TypeAdapter
from segments.typing import LabelAttributes, Link, TaskAttributes


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
