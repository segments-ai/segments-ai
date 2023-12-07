# Types

```{seealso}
Please visit our [docs](https://docs.segments.ai/) for more information on Segments.ai and visit the [setup page](https://sdkdocs.segments.ai/en/latest/setup.html) to learn how to install and setup the Segments.ai Python SDK.

```

## Datasets

```{eval-rst}
.. autopydantic_model:: segments.typing.Dataset

.. autopydantic_model:: segments.typing.Collaborator

.. autopydantic_model:: segments.typing.User

.. autopydantic_model:: segments.typing.Subscription

.. autoclass:: segments.typing.TaskType
    :members:
    :undoc-members:
```

## Samples

```{eval-rst}
.. autopydantic_model:: segments.typing.Sample

.. autopydantic_model:: segments.typing.SampleAttributes

.. autopydantic_model:: segments.typing.Issue

.. autopydantic_model:: segments.typing.IssueComment

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

.. autopydantic_model:: segments.typing.MultiSensorPointcloudSequenceSampleAttributes

.. autopydantic_model:: segments.typing.MultiSensorImageSequenceSampleAttributes
```

### Text sample

```{eval-rst}
.. autopydantic_model:: segments.typing.TextSampleAttributes
```

## Labels

```{eval-rst}
.. autopydantic_model:: segments.typing.Label

.. autopydantic_model:: segments.typing.LabelAttributes

.. autopydantic_model:: segments.typing.LabelSummary

.. autoclass:: segments.typing.LabelStatus
    :members:
    :undoc-members:
```

### Image segmentation label

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSegmentationLabelAttributes

.. autopydantic_model:: segments.typing.Annotation
```

### Image sequence segmentation label

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSequenceSegmentationLabelAttributes

.. autopydantic_model:: segments.typing.ImageSegmentationFrame
    :show-inheritance:
```

### Image vector label

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageVectorLabelAttributes

.. autopydantic_model:: segments.typing.ImageVectorAnnotation

.. autoclass:: segments.typing.ImageVectorAnnotationType
    :members:
    :undoc-members:
```

### Image sequence vector label

```{eval-rst}
.. autopydantic_model:: segments.typing.ImageSequenceVectorLabelAttributes

.. autopydantic_model:: segments.typing.ImageVectorFrame
    :show-inheritance:

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

.. autopydantic_model:: segments.typing.PointcloudCuboidAnnotation

.. autoclass:: segments.typing.PointcloudCuboidAnnotationType
    :members:
    :undoc-members:
```

### Point cloud vector label

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudVectorLabelAttributes

.. autopydantic_model:: segments.typing.PointcloudVectorAnnotation

.. autoclass:: segments.typing.PointcloudVectorAnnotationType
    :members:
    :undoc-members:
```

### Point cloud sequence segmentation

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceSegmentationLabelAttributes

.. autopydantic_model:: segments.typing.PointcloudSegmentationFrame
    :show-inheritance:

.. autopydantic_model:: segments.typing.PointcloudSequenceSegmentationAnnotation
    :show-inheritance:
```

### Point cloud sequence cuboid label

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidLabelAttributes

.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidFrame
    :show-inheritance:

.. autopydantic_model:: segments.typing.PointcloudSequenceCuboidAnnotation
    :show-inheritance:
```

### Point cloud sequence vector label

```{eval-rst}
.. autopydantic_model:: segments.typing.PointcloudSequenceVectorLabelAttributes

.. autopydantic_model:: segments.typing.PointcloudSequenceVectorFrame
    :show-inheritance:

.. autopydantic_model:: segments.typing.PointcloudSequenceVectorAnnotation
    :show-inheritance:
```

### Multi-sensor label

```{eval-rst}
.. autopydantic_model:: segments.typing.MultiSensorLabelAttributes

.. autopydantic_model:: segments.typing.MultiSensorPointcloudSequenceCuboidLabelAttributes

.. autopydantic_model:: segments.typing.MultiSensorImageSequenceVectorLabelAttributes
```

### Text label

```{eval-rst}
.. autopydantic_model:: segments.typing.TextLabelAttributes

.. autopydantic_model:: segments.typing.TextAnnotation
```

## Labelsets

```{eval-rst}
.. autopydantic_model:: segments.typing.Labelset
```

## Releases

```{eval-rst}
.. autopydantic_model:: segments.typing.Release

.. autoclass:: segments.typing.ReleaseStatus
    :members:
    :undoc-members:
```

## Files

```{eval-rst}
.. autopydantic_model:: segments.typing.File

.. autopydantic_model:: segments.typing.PresignedPostFields
```

## Helper classes

```{eval-rst}
.. autopydantic_model:: segments.typing.URL

.. autopydantic_model:: segments.typing.Owner

.. autopydantic_model:: segments.typing.TaskAttributes

.. autopydantic_model:: segments.typing.TaskAttributeCategory

.. autoclass:: segments.typing.SelectTaskAttribute
    :members:
    :undoc-members:

.. autoclass:: segments.typing.TextTaskAttribute
:members:
:undoc-members:

.. autoclass:: segments.typing.NumberTaskAttribute
:members:
:undoc-members:

.. autoclass:: segments.typing.CheckboxTaskAttribute
:members:
:undoc-members:

.. autopydantic_model:: segments.typing.LabelStats

.. autopydantic_model:: segments.typing.XYZ

.. autopydantic_model:: segments.typing.XYZW

.. autopydantic_model:: segments.typing.EgoPose

.. autopydantic_model:: segments.typing.PCD

.. autoclass:: segments.typing.PCDType
:members:
:undoc-members:

.. autopydantic_model:: segments.typing.CalibratedImage
:show-inheritance:

.. autoclass:: segments.typing.CameraConvention
:members:
:undoc-members:

.. autopydantic_model:: segments.typing.CameraIntrinsics

.. autopydantic_model:: segments.typing.CameraExtrinsics

.. autopydantic_model:: segments.typing.Distortion

.. autoclass:: segments.typing.CameraDistortionModel
:members:
:undoc-members:

.. autopydantic_model:: segments.typing.FisheyeDistortionCoefficients

.. autopydantic_model:: segments.typing.BrownConradyDistortionCoefficients

.. autoclass:: segments.typing.Category
:members:
:undoc-members:

.. autoclass:: segments.typing.Role
:members:
:undoc-members:

.. py:data:: segments.typing.ImageAttributes

.. py:data:: segments.typing.ObjectAttributes

.. py:data:: segments.typing.FormatVersion

.. py:data:: segments.typing.RGB

.. py:data:: segments.typing.RGBA

.. autoclass:: segments.typing.ExportFormat
:members:
:undoc-members:
```
