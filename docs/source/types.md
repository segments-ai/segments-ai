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

### Text sample

```{eval-rst}
.. autopydantic_model:: segments.typing.TextSampleAttributes
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
```

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSequenceVectorAnnotation
    :show-inheritance:
```

### Point cloud segmentation label

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

### Point cloud sequence segmentation

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceSegmentationLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSegmentationFrame
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceSegmentationAnnotation
```

### Point cloud sequence cuboid label

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidFrame
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidAnnotation
    :show-inheritance:
```

### Point cloud sequence vector label

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceVectorLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceVectorFrame
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
.. autopydantic_model:: segments.typing.MultiSensorImageSequenceVectorLabelAttributes
```

### Text label

```{eval-rst}
.. autopydantic_model:: segments.typing.TextLabelAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.TextAnnotation
```

## Labelsets

```{eval-rst}
.. autopydantic_model:: segments.typing.Labelset
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

## Helper classes

```{eval-rst}
.. autopydantic_model:: segments.typing.URL
```

```{eval-rst}
.. autopydantic_model:: segments.typing.Owner
```

```{eval-rst}
.. autopydantic_model:: segments.typing.TaskAttributes
```

```{eval-rst}
.. autopydantic_model:: segments.typing.TaskAttributeCategory
```

```{eval-rst}
.. autoclass:: segments.typing.SelectTaskAttribute
    :members:
    :undoc-members:
```

```{eval-rst}
.. autoclass:: segments.typing.TextTaskAttribute
    :members:
    :undoc-members:
```

```{eval-rst}
.. autoclass:: segments.typing.NumberTaskAttribute
    :members:
    :undoc-members:
```

```{eval-rst}
.. autoclass:: segments.typing.CheckboxTaskAttribute
    :members:
    :undoc-members:
```

```{eval-rst}
.. autopydantic_model:: segments.typing.LabelStats
```

```{eval-rst}
.. autopydantic_model:: segments.typing.Statistics
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
.. autopydantic_model:: segments.typing.Distortion
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
