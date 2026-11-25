"""Compressed form of ``point_annotations``, the per-point id array of point cloud segmentation labels.

The API stores the array as ``{"encoding": "gzip-uint8-base64", "data": "..."}``: the ids as little-endian unsigned
integers (8 or 16 bits), gzipped and base64 encoded. The client converts it to and from a list of ints for you, so you
only need these functions when you call the API directly or want a numpy array.
"""

from __future__ import annotations

import base64
import gzip
from typing import Any, Dict, Iterator, Mapping, Sequence, Union

import numpy as np
import numpy.typing as npt


# Sent with every request to tell the API this client understands the compressed form
POINT_ENCODING_HEADER = "Segments-Point-Encoding"
POINT_ENCODING = "gzip-base64"

# Encoding name -> dtype, smallest first
_DTYPES = {"gzip-uint8-base64": np.dtype("<u1"), "gzip-uint16-base64": np.dtype("<u2")}


def encode_point_annotations(point_annotations: Union[Sequence[int], npt.NDArray[Any]]) -> Dict[str, str]:
    """Compress a point annotations array.

    Args:
        point_annotations: The object id of each point (0 for unlabeled points).

    Returns:
        The compressed point annotations: a dict with an ``encoding`` and ``data``.

    Raises:
        :exc:`ValueError`: If the array holds something else than integers from 0 to 65535.
    """
    ids = np.asarray(point_annotations)
    if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
        raise ValueError("point_annotations must be a one-dimensional array of integers.")
    if ids.size and (ids.min() < 0 or ids.max() > 65535):
        raise ValueError("point_annotations must hold integers from 0 to 65535.")
    encoding = "gzip-uint8-base64" if not ids.size or ids.max() <= 255 else "gzip-uint16-base64"
    data = gzip.compress(ids.astype(_DTYPES[encoding]).tobytes(), compresslevel=6, mtime=0)
    return {"encoding": encoding, "data": base64.b64encode(data).decode("ascii")}


def decode_point_annotations(encoded: Mapping[str, Any]) -> npt.NDArray[Any]:
    """Decompress point annotations.

    Args:
        encoded: The compressed point annotations: a dict with an ``encoding`` and ``data``.

    Returns:
        The object id of each point, as a numpy array.

    Raises:
        :exc:`ValueError`: If the encoding is unknown (upgrade the SDK) or the data is invalid.
    """
    encoding = encoded.get("encoding")
    dtype = _DTYPES.get(encoding)  # type: ignore[arg-type]
    if dtype is None:
        raise ValueError(
            f"Unknown point_annotations encoding {encoding!r}. Please upgrade by running `pip install --upgrade segments-ai`."
        )
    try:
        raw = gzip.decompress(base64.b64decode(encoded["data"], validate=True))
        return np.frombuffer(raw, dtype=dtype)
    except (KeyError, TypeError, ValueError, OSError, EOFError) as e:
        raise ValueError("Invalid point_annotations data.") from e


def _frames(attributes: Any) -> Iterator[Dict[str, Any]]:
    """Yield every dict of the label attributes that can hold ``point_annotations``: the attributes themselves for a
    single-frame label, each frame of a sequence, and the same for each sensor of a multi-sensor label."""
    if not isinstance(attributes, dict):
        return
    sensors = attributes.get("sensors")
    if isinstance(sensors, list):
        for sensor in sensors:
            if isinstance(sensor, dict):
                yield from _frames(sensor.get("attributes"))
        return
    frames = attributes.get("frames")
    if isinstance(frames, list):
        yield from (frame for frame in frames if isinstance(frame, dict))
    else:
        yield attributes


def compress_label_attributes(attributes: Dict[str, Any]) -> None:
    """Replace every ``point_annotations`` array in the label attributes with its compressed form, in place.

    Arrays that cannot be compressed (e.g. with ids above 65535) are left as they are.
    """
    for frame in _frames(attributes):
        value = frame.get("point_annotations")
        if isinstance(value, (list, np.ndarray)) and len(value):
            try:
                frame["point_annotations"] = encode_point_annotations(value)
            except ValueError:
                pass
