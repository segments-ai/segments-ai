```{seealso}
Please visit our [docs](https://docs.segments.ai/) for more information on Segments.ai and visit the [setup page](https://sdkdocs.segments.ai/en/latest/setup.html) to learn how to install and setup the Segments.ai Python SDK.

```

# Utils

## Load

### Load an image from a url

```{eval-rst}
.. autofunction:: segments.utils.load_image_from_url
```

### Load a label bitmap from a url

```{eval-rst}
.. autofunction:: segments.utils.load_label_bitmap_from_url
```

### Load a release

```{eval-rst}
.. autofunction:: segments.utils.load_release
```

### Load a release as a model

```{eval-rst}
.. autofunction:: segments.utils.load_release_as_model
```

## Transform

### Turn a bitmap into a file

```{eval-rst}
.. autofunction:: segments.utils.bitmap2file
```

### Turn an instance bitmap and annotations dict into a semantic bitmap

```{eval-rst}
.. autofunction:: segments.utils.get_semantic_bitmap
```

### Fix an image rotation

```{eval-rst}
.. autofunction:: segments.utils.handle_exif_rotation
```

## Export

### Export a dataset to a different format

```{eval-rst}
.. autofunction:: segments.utils.export_dataset
```

## Show

### Show the exported contours of a segmented image

```{eval-rst}
.. autofunction:: segments.utils.show_polygons
```
