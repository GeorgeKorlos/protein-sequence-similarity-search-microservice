import os
from google.cloud import storage

uri = os.environ["GCS_INDEX_URI"]
assert uri.startswith("gs://"), f"Invalid GCS URI: {uri}"
parts = uri[5:].split("/", 1)
bucket_name = parts[0]
prefix = parts[1] if len(parts) > 1 else ""

client = storage.Client()
blobs = list(client.list_blobs(bucket_name, prefix=prefix))
assert blobs, f"No files found at {uri}"

for blob in blobs:
    filename = os.path.basename(blob.name)
    dest = f"/tmp/index/{filename}"
    print(f"  {blob.name} -> {dest}")
    blob.download_to_filename(dest)

print("Index download complete.")
