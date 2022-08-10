# Setup

```{seealso}
Please refer to the [Python SDK quickstart](https://docs.segments.ai/tutorials/python-sdk-quickstart) for a full example of working with the Python SDK.
```

First install the SDK.

```{code-block} bash
pip install --upgrade segments-ai
```

Import the `segments` package in your python file and and set up a client with an API key. An API key can be created on your [user account page](https://segments.ai/account).

```{code-block} python
from segments import SegmentsClient

api_key = "YOUR_API_KEY"
client = SegmentsClient(api_key)
```

Or store your Segments API key in your environment (`SEGMENTS_API_KEY = YOUR_API_KEY`):

```{code-block} python
from segments import SegmentsClient

client = SegmentsClient()
```

You can also use the client as a context manager:

```{code-block} python
with SegmentsClient() as client:
    client.get_datasets()
```
