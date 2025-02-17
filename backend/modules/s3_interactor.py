import boto3
import json

def create_directory(bucket: str, directory: str):
    """creates a given directory in the bucket. """
    s3_client = boto3.client("s3")
    s3_client.put_object(Bucket=bucket, Key=directory)

def read_pdf(bucket: str, directory: str) -> bytes:
    """reads a pdf file from the bucket"""
    if len(directory) == 0:
        return None
    
    s3_client = boto3.client("s3")
    try:
        response = s3_client.get_object(Bucket = bucket, Key = directory)
        return response["Body"].read()
    except s3_client.exceptions.NoSuchKey:
        return None

def read_json(bucket: str, directory: str) -> str:
    s3_client = boto3.client("s3")
    try:
        response = s3_client.get_object(Bucket = bucket, Key = directory)
        file_contents = response["Body"].read().decode("utf-8")
        return json.loads(file_contents)
    except s3_client.exceptions.NoSuchKey:
        return {}
    
def save_as_json(bucket: str, directory: str, data: str) -> None:
    s3_client = boto3.client("s3")
    json_data = json.dumps(data)
    s3_client.put_object(Bucket=bucket, Key=directory, Body=json_data)