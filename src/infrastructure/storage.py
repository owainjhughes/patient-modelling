import os
from minio import Minio


class MinIOStorage:

    def __init__(self):
        self.client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False
        )
    
    def upload_file(self, bucket, object_key, file_data, length):
        self.client.put_object(bucket, object_key, file_data, length)

    def download_file(self, bucket, object_key):
        return self.client.get_object(bucket, object_key)