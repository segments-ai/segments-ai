from segments.typing import TaskAttributes


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
