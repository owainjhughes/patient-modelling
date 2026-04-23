#claude generated test
import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from infrastructure.storage import MinIOStorage

BUCKET = "dsapi"
OBJECT_KEY = "uploads/test.csv"
TEST_FILE = os.path.join(os.path.dirname(__file__), "src", "app", "data", "raw_data.csv")


def main():
    storage = MinIOStorage()

    # Ensure bucket exists
    if not storage.client.bucket_exists(BUCKET):
        storage.client.make_bucket(BUCKET)
        print(f"Created bucket: {BUCKET}")
    else:
        print(f"Bucket already exists: {BUCKET}")

    # Upload
    with open(TEST_FILE, "rb") as f:
        file_bytes = f.read()

    file_size = len(file_bytes)
    storage.upload_file(BUCKET, OBJECT_KEY, io.BytesIO(file_bytes), file_size)
    print(f"Uploaded {TEST_FILE} → {OBJECT_KEY} ({file_size} bytes)")

    # Download and verify
    response = storage.download_file(BUCKET, OBJECT_KEY)
    data = response.read(200)
    response.close()
    response.release_conn()

    print(f"\nFirst 200 bytes of downloaded file:\n{data.decode('utf-8', errors='replace')}")
    print("\nTest passed.")


if __name__ == "__main__":
    main()
