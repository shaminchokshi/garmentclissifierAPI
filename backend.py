# from fastapi import FastAPI
# from typing import List
# import boto3
# import os
# app = FastAPI()

# # Configure your AWS credentials
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#     region_name=os.getenv('AWS_DEFAULT_REGION')
# )
# # Define your S3 bucket name
# BUCKET_NAME = 'indo-fashion-dataset'

# def get_similar_images(category: str, n: int = 2) -> List[str]:
#     response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f'clothes/train/{category}/')
#     all_images = [obj['Key'] for obj in response.get('Contents', [])]
#     return all_images[:n]

# @app.post("/get_similar_images/")
# async def get_similar_images_endpoint(category: str):
#     similar_images = get_similar_images(category)
#     image_urls = [f'https://{BUCKET_NAME}.s3.amazonaws.com/{img}' for img in similar_images]
#     return {"image_urls": image_urls}



#-----------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import boto3
import os
from dotenv import load_dotenv

app = FastAPI()

# Load environment variables from .env file
load_dotenv()

# Configure your AWS credentials
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

# Define your S3 bucket name
BUCKET_NAME = 'indo-fashion-dataset'

class ImageRequest(BaseModel):
    category: str

def get_similar_images(category: str, n: int = 2) -> List[str]:
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=f'clothes/train/{category}/')
    if 'Contents' not in response:
        raise HTTPException(status_code=404, detail="Category not found or no images in category")
    all_images = [obj['Key'] for obj in response['Contents']]
    return all_images[:n]

@app.post("/get_similar_images/")
async def get_similar_images_endpoint(request: ImageRequest):
    similar_images = get_similar_images(request.category)
    image_urls = [f'https://{BUCKET_NAME}.s3.amazonaws.com/{img}' for img in similar_images]
    return {"image_urls": image_urls}
