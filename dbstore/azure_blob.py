from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
import os

# Replace with your connection string
AZURE_CONNECTION_STRING = "your_connection_string"
CONTAINER_NAME = "your-container"
BLOB_NAME = "example.txt"
LOCAL_FILE_PATH = "example.txt"


def main():
    try:
        # Create blob service client
        blob_service_client = BlobServiceClient.from_connection_string(
            AZURE_CONNECTION_STRING
        )

        # Get container client
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        # Create container if it does not exist
        try:
            container_client.create_container()
            print(f"Container '{CONTAINER_NAME}' created.")
        except Exception:
            print(f"Container '{CONTAINER_NAME}' already exists.")

        # Upload file
        with open(LOCAL_FILE_PATH, "rb") as data:
            container_client.upload_blob(name=BLOB_NAME, data=data, overwrite=True)
            print(f"Uploaded {LOCAL_FILE_PATH} as {BLOB_NAME}.")

        # List blobs in container
        print("Listing blobs...")
        blobs = container_client.list_blobs()
        for blob in blobs:
            print(f" - {blob.name}")

        # Download blob
        download_path = os.path.join("downloaded_" + BLOB_NAME)
        with open(download_path, "wb") as f:
            download_stream = container_client.download_blob(BLOB_NAME)
            f.write(download_stream.readall())
            print(f"Downloaded blob to {download_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
