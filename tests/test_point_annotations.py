import copy
import json

import numpy as np
import pytest
from pydantic import TypeAdapter
from segments.client import SegmentsClient
from segments.point_annotations import (
    POINT_ENCODING_HEADER,
    compress_label_attributes,
    decode_point_annotations,
    encode_point_annotations,
)
from segments.typing import (
    Label,
    LabelAttributes,
    PointcloudSegmentationLabelAttributes,
    PointcloudSequenceSegmentationLabelAttributes,
)


def sequence(*frames):  # type: ignore[no-untyped-def]
    return {"format_version": "0.2", "frames": [{"annotations": [], "point_annotations": frame} for frame in frames]}


@pytest.mark.parametrize(
    "ids, encoding",
    [([0, 1, 255, 2], "gzip-uint8-base64"), ([0, 256, 65535], "gzip-uint16-base64"), ([], "gzip-uint8-base64")],
)
def test_encode_decode_round_trip(ids, encoding) -> None:  # type: ignore[no-untyped-def]
    for value in (ids, np.array(ids, dtype=np.int64)):
        encoded = encode_point_annotations(value)
        assert encoded["encoding"] == encoding
        assert json.loads(json.dumps(encoded)) == encoded
        assert decode_point_annotations(encoded).tolist() == ids


def test_decode_matches_the_wire_format() -> None:
    # gzip of the bytes 03 00 01, and of the little-endian uint16 values 300 and 2: as encoded by the API
    assert decode_point_annotations(
        {"encoding": "gzip-uint8-base64", "data": "H4sIAAAAAAAA/2NmYAQA3VcAigMAAAA="}
    ).tolist() == [3, 0, 1]
    assert decode_point_annotations(
        {"encoding": "gzip-uint16-base64", "data": "H4sIAAAAAAAA/9NhZGIAAC/HVPgEAAAA"}
    ).tolist() == [300, 2]


@pytest.mark.parametrize("ids", [[0, None], [0, 65536], [0, -1], [0.5], [[1, 2]]])
def test_encode_rejects_what_does_not_fit(ids) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        encode_point_annotations(ids)


@pytest.mark.parametrize(
    "encoded",
    [
        {"encoding": "zstd", "data": ""},
        {"encoding": "gzip-uint8-base64"},
        {"encoding": "gzip-uint8-base64", "data": "x"},
    ],
)
def test_decode_rejects_invalid_values(encoded) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        decode_point_annotations(encoded)


def test_label_with_compressed_point_annotations_validates_to_lists() -> None:
    compressed = sequence(encode_point_annotations([0, 1, 1]), encode_point_annotations([300, 0]))
    attributes = TypeAdapter(LabelAttributes).validate_python(compressed)
    assert isinstance(attributes, PointcloudSequenceSegmentationLabelAttributes)
    assert [frame.point_annotations for frame in attributes.frames] == [[0, 1, 1], [300, 0]]
    # Dumping gives the dense arrays, like before
    assert attributes.model_dump(mode="json", exclude_unset=True) == sequence([0, 1, 1], [300, 0])

    label = Label.model_validate(
        {
            "sample_uuid": "uuid",
            "label_type": "pointcloud-segmentation",
            "label_status": "LABELED",
            "labelset": "ground-truth",
            "attributes": {"annotations": [], "point_annotations": encode_point_annotations([2, 2])},
            "created_at": "2026-01-01",
            "created_by": "jane",
            "updated_at": "2026-01-01",
        }
    )
    assert label.attributes.point_annotations == [2, 2]  # type: ignore[union-attr]


def test_dense_and_numpy_point_annotations_still_validate() -> None:
    for value in ([0, 1], np.array([0, 1])):
        attributes = PointcloudSegmentationLabelAttributes.model_validate(
            {"annotations": [], "point_annotations": value}
        )
        assert attributes.point_annotations == [0, 1]
    with pytest.raises(ValueError):
        PointcloudSegmentationLabelAttributes.model_validate({"annotations": [], "point_annotations": ["a"]})
    with pytest.raises(ValueError):
        PointcloudSegmentationLabelAttributes.model_validate(
            {"annotations": [], "point_annotations": {"encoding": "zstd"}}
        )


def test_compress_label_attributes() -> None:
    single = {"annotations": [], "point_annotations": [1, 2]}
    multi = {
        "sensors": [
            {"name": "camera", "task_type": "image-vector-sequence", "attributes": []},
            {"name": "lidar", "task_type": "pointcloud-segmentation-sequence", "attributes": sequence([1, 2])},
        ]
    }
    for attributes in (single, sequence([1, 2]), multi):
        compressed = copy.deepcopy(attributes)
        compress_label_attributes(compressed)
        assert json.dumps(compressed) == json.dumps(attributes).replace(
            "[1, 2]", json.dumps(encode_point_annotations([1, 2]))
        )

    # Left as they are: empty, ids that don't fit, no point annotations
    for attributes in (sequence([], [0, 65536], [0, None]), {"annotations": []}):
        unchanged = copy.deepcopy(attributes)
        compress_label_attributes(unchanged)
        assert unchanged == attributes


def test_client_announces_the_encoding() -> None:
    client = SegmentsClient.__new__(SegmentsClient)
    client.api_key = "key"
    assert client._get_headers()[POINT_ENCODING_HEADER] == "gzip-base64"
