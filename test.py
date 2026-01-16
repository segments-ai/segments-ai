from segments import SegmentsClient
import json
import sys
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# You can find your api key at https://segments.ai/account
# api_key = ""
api_key = ""

client = SegmentsClient(api_key, api_url="http://localhost:5000/")

segments_dataset_name = "admin-antoine/flathead-clone-20"
all_segments_samples = {}
page = 2

try:
    print(f"Calling get_samples for page {page}...")
    start_time = time.time()

    samples = client.get_samples(
        segments_dataset_name,
        page=page,
        per_page=3,
        labelset="ground-truth",
        include_full_label=True,
        label_status=['REVIEWED'],
    )

    sdk_call_time = time.time() - start_time
    print(f"SDK call completed in {sdk_call_time:.2f} seconds")

    print(f"Page {page}: Retrieved {len(samples)} samples")

    samples_json = json.dumps([s.model_dump(mode="json") for s in samples])
    body_size_bytes = len(samples_json.encode('utf-8'))
    body_size_mb = body_size_bytes / (1024 * 1024)
    print(f"Response body size: {body_size_bytes:,} bytes ({body_size_mb:.2f} MB)")

    for sample in samples:
        all_segments_samples[sample.uuid] = sample
    page += 1

    # Log samples info at the end
    print("\n=== Samples fetched ===")
    for i, sample in enumerate(samples[:3]):  # Log first 3 samples
        print(f"\nSample {i+1}:")
        print(f"  UUID: {sample.uuid}")
        print(f"  Name: {sample.name}")
        print(f"  Attributes type: {type(sample.attributes)}")
        print(f"  Attributes: {sample.attributes}")

except NotFoundError:
    pass
