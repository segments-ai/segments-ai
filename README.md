# Python SDK

## Setup

First install the SDK.

```bash
pip install segments-ai --upgrade
```

Import the `segments` package in your python file and and set up a client with an API key.

```python
from segments import SegmentsClient

api_key = "eabdde840de8c8853329c086bc4165591cb3d74c"
client = SegmentsClient(api_key)
```

## Add a sample to a dataset

```python
dataset = "jane/flowers"
name = "violet.jpg"
attributes = {
    "image": {
        "url": "https://example.com/violet.jpg"
    }
}

sample = client.add_sample(dataset, name, attributes)
print(sample)
```

If the image is on your local computer, you should first upload it to a cloud storage service like Amazon S3, Google Cloud Storage, Imgur, or [our asset storage service](python-sdk.md#upload-a-file-as-an-asset).

## Get a sample by id

```python
sample = client.get_sample(uuid="602a3eec-a61c-4a77-9fcc-3037ce5e9606")
print(sample)
```

## Add a label to a sample

A label can be added to a sample in relation to a _task_ defined on the dataset, such as an image classification task or an image segmentation task.

The task type specifies the format that is required for the label: for a classification task, the attributes field should contain different fields than for a segmentation task.

```python
sample_uuid = "602a3eec-a61c-4a77-9fcc-3037ce5e9606"
task_name = "segmentation"
attributes = {
    "segmentation_bitmap_url": "https://segmentsai-prod.s3.eu-west-2.amazonaws.com/assets/bert/49f6aa10-8967-4305-985c-cdc1e8f89b93.png" 
    "objects": [
        {
            "instance_id": 1
            "category_id": 2
        },
        {
            "instance_id": 2
            "category_id": 3
        }
    ]
}

client.add_label(sample_uuid, task_name, attributes)
```

## Get all samples in a dataset

```python
dataset = "jane/flowers"
samples = client.get_samples(dataset)

for sample in samples:
    print(sample["name"], sample["uuid"])
```

## Upload a file as an asset

Example with an image:

```python
from PIL import Image
from io import BytesIO

image = Image.open("/home/jane/flowers/violet.jpg")
file = BytesIO()
image.save(file, 'PNG')

asset = client.upload_asset(file, filename="violet.jpg")
image_url = asset["url"]
```

