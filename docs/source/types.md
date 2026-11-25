---
tocdepth: 3
---
# Types

```{seealso}
Please visit our [docs](https://docs.segments.ai/) for more information on Segments.ai and visit the [setup page](https://sdkdocs.segments.ai/en/latest/setup.html) to learn how to install and setup the Segments.ai Python SDK.

```

## Datasets

```{eval-rst}
.. autopydantic_model:: segments.typing.Dataset
```

```{eval-rst}
.. autopydantic_model:: segments.typing.Collaborator
```


```{eval-rst}
.. autopydantic_model:: segments.typing.User
```

```{eval-rst}
.. autoclass:: segments.typing.TaskType
    :members:
    :undoc-members:
```

## Samples

```{eval-rst}
.. autopydantic_model:: segments.typing.Sample
```

```{eval-rst}
.. autopydantic_model:: segments.typing.Issue
```

```{eval-rst}
.. autopydantic_model:: segments.typing.IssueComment
```

```{eval-rst}
.. autoclass:: segments.typing.IssueStatus
    :members:
    :undoc-members:
```

### Image sample

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSampleAttributes
```

### Image sequence sample

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSequenceSampleAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageFrame
    :show-inheritance:
```

### Point cloud sample

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSampleAttributes
```

### Point cloud sequence sample

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceSampleAttributes
```

### Multi-sensor sample

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorSampleAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorPointcloudSequenceSampleAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorImageSequenceSampleAttributes
```

## Labels

```{eval-rst}
.. autopydantic_model:: segments.typing.Label
```

```{eval-rst}
.. autoclass:: segments.typing.LabelStatus
    :members:
    :undoc-members:
```

### Image segmentation label

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSegmentationLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.Annotation
```

### Image vector label

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageVectorLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageVectorAnnotation
```

```{eval-rst}
.. autoclass:: segments.typing.ImageVectorAnnotationType
    :members:
    :undoc-members:
```

### Image sequence vector label

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSequenceVectorLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageVectorFrame
    :show-inheritance:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSequenceVectorAnnotation
    :show-inheritance:
```

### Point cloud segmentation label

`point_annotations` holds the object id of each point (0 for unlabeled points), in the order of the points in the point cloud. This also applies to each frame of a point cloud sequence segmentation label.

````{note}
The API stores `point_annotations` compressed, because a plain JSON array makes labels of long sequences very large. The compressed form is an object instead of an array:

```json
"point_annotations": {"encoding": "gzip-uint8-base64", "data": "H4sIA..."}
```

`data` holds the ids as little-endian unsigned integers (`uint8`, or `uint16` when an id is above 255), gzipped and base64 encoded.

The SDK converts this for you: `point_annotations` is always a list of ints when you read a label, and you can pass a list or a numpy array when you add or update one. You only meet the compressed form when you call the API directly and send the `Segments-Point-Encoding: gzip-base64` request header. Without that header the API returns and accepts plain arrays, as do webhooks and release files. Use [`encode_point_annotations` and `decode_point_annotations`](utils.md) to convert it yourself.
````

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSegmentationLabelAttributes
```

### Point cloud cuboid label

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudCuboidLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudCuboidAnnotation
```

```{eval-rst}
.. autoclass:: segments.typing.PointcloudCuboidAnnotationType
    :members:
    :undoc-members:
```

```{eval-rst}
.. autoclass:: segments.typing.PointcloudCuboidAnnotationTypeLiteral
```

### Point cloud vector label

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudVectorLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudVectorAnnotation
```

```{eval-rst}
.. autoclass:: segments.typing.PointcloudVectorAnnotationType
    :members:
    :undoc-members:
```

```{eval-rst}
.. autoclass:: segments.typing.PointcloudVectorAnnotationTypeLiteral
```

### Point cloud sequence segmentation

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceSegmentationLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSegmentationFrame
    :show-inheritance:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceSegmentationAnnotation
    :show-inheritance:
```

### Point cloud sequence cuboid label


```{eval-rst}
.. autopydantic_model:: segments.typing.BasePointcloudSequenceCuboidLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidLabelAttributes
    :show-inheritance:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidFrame
    :show-inheritance:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidAnnotation
    :show-inheritance:
```

### Point cloud sequence vector label


```{eval-rst}
.. autopydantic_model:: segments.typing.BasePointcloudSequenceVectorLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceVectorLabelAttributes
    :show-inheritance:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceVectorFrame
    :show-inheritance:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceVectorAnnotation
    :show-inheritance:
```

### Multi-sensor label

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorPointcloudSequenceCuboidLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorPointcloudSequenceVectorLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorImageSequenceVectorLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorImageSequenceSegmentationLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorPointcloudSequenceSegmentationLabelAttributes
```

### Links

```{eval-rst}
.. autopydantic_model:: segments.typing.Link
```

## Labelsets

```{eval-rst}
.. autopydantic_model:: segments.typing.Labelset
```

## Workunits

```{eval-rst}
.. autopydantic_model:: segments.typing.Workunit
```

## Releases

```{eval-rst}
.. autopydantic_model:: segments.typing.Release
```

```{eval-rst}
.. autoclass:: segments.typing.ReleaseStatus
    :members:
    :undoc-members:
```

## Files

```{eval-rst}
.. autopydantic_model:: segments.typing.File
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PresignedPostFields
```

## Object attribute definitions


### Base task attributes

This types are not used directly, but are instead mix-ins for their respective object and linking attribute classes.

```{eval-rst}
.. autopydantic_model:: segments.typing.BaseAttribute
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BaseSelectTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BaseTextTaskAttribute
    :show-inheritance: BaseModel
```


```{eval-rst}
.. autopydantic_model:: segments.typing.BaseNumberTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BaseCheckboxTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BaseMultiselectTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BaseVector3TaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BaseQuaternionTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BasePointsTaskAttribute
    :show-inheritance: BaseModel
```

### Task attributes

These types are used for defining task attributes.

```{eval-rst}
.. autopydantic_model:: segments.typing.TaskAttributes
```


```{eval-rst}
.. autoclass:: segments.typing.TaskAttribute
```

```{eval-rst}
.. autopydantic_model:: segments.typing.TaskAttributeCategory
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BaseTaskAttribute
```

```{eval-rst}
.. autopydantic_model:: segments.typing.SelectTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.TextTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.NumberTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.CheckboxTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiselectTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.Vector3TaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.QuaternionTaskAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointsTaskAttribute
    :show-inheritance: BaseModel
```

### Link attributes

These types are used for defining link attributes.


```{eval-rst}
.. autopydantic_model:: segments.typing.BaseTaskLinkAttribute
```


```{eval-rst}
.. autoclass:: segments.typing.TaskLinkAttribute
```

```{eval-rst}
.. autopydantic_model:: segments.typing.CheckboxTaskLinkAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.NumberTaskLinkAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.TextTaskLinkAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiselectTaskLinkAttribute
    :show-inheritance: BaseModel
```

```{eval-rst}
.. autopydantic_model:: segments.typing.SelectTaskLinkAttribute
    :show-inheritance: BaseModel
```

## Helper classes

```{eval-rst}
.. autopydantic_model:: segments.typing.URL
```

```{eval-rst}
.. autopydantic_model:: segments.typing.Owner
```


```{eval-rst}
.. autopydantic_model:: segments.typing.LabelStats
```

```{eval-rst}
.. autopydantic_model:: segments.typing.XYZ
```

```{eval-rst}
.. autopydantic_model:: segments.typing.XYZW
```

```{eval-rst}
.. autopydantic_model:: segments.typing.EgoPose
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PCD
```

```{eval-rst}
.. autoclass:: segments.typing.PCDType
    :members:
    :undoc-members:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.CalibratedImage
    :show-inheritance:
```

```{eval-rst}
.. autoclass:: segments.typing.CameraConvention
    :members:
    :undoc-members:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.CameraIntrinsics
```

```{eval-rst}
.. autopydantic_model:: segments.typing.CameraExtrinsics
```

```{eval-rst}
.. autopydantic_model:: segments.typing.DistortionFisheye
```

```{eval-rst}
.. autopydantic_model:: segments.typing.DistortionBrownConrady
```

```{eval-rst}
.. autoclass:: segments.typing.CameraDistortionModel
    :members:
    :undoc-members:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.FisheyeDistortionCoefficients
```

```{eval-rst}
.. autopydantic_model:: segments.typing.BrownConradyDistortionCoefficients
```
