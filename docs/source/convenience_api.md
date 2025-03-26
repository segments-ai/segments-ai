# High-level API

## Usage

There are 2 main approaches to using the SDK. One is using the [SegmentsClient](client.md) for each API call directly. However, you can also methods attached to the segments.ai resources directly. We recommend using the latter in most cases, as this allows the SDK to perform better validation checks to the inputs you provide to the API. This will both speed up performance, reduce the risk of errors, and give more descriptive error messages in case there is a mistake anywhere.

When using the resource-driven approach, you will always start out fetching or creating a resource using [SegmentsClient](client.md). 

```python
import segments

client = segment.SegmentsClient(api_key_here)
dataset = client.get_dataset(your_dataset_identifier)

# Get all samples in a dataset
samples = dataset.get_samples()
# Add samples to the dataset
sample = dataset.add_sample(attributes=your_sample_attributes, name="Sample 1")
# Add a prelabel to that sample
label = sample.add_label(attributes=your_prelabel_attributes)

# Update the sample
sample.update(name="New name")
# Delete the sample
sample.delete()
```


## API

```{eval-rst}
.. autoclass:: segments.convenience_api.Dataset
    :show-inheritance:
    :members:
    :undoc-members:
    :exclude-members: model_config, model_post_init
```

```{eval-rst}
.. autoclass:: segments.convenience_api.Sample
    :show-inheritance:
    :members:
    :undoc-members:
    :exclude-members: model_config, model_post_init
```

```{eval-rst}
.. autoclass:: segments.convenience_api.Label
    :show-inheritance:
    :members:
    :undoc-members:
    :exclude-members: model_config, model_post_init
```
