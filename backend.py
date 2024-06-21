# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from typing import List
# import boto3
# import os
# from dotenv import load_dotenv
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from pinecone import Pinecone, ServerlessSpec
# import base64
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import img_to_array

# app = FastAPI()

# # Load environment variables from .env file
# load_dotenv()

# # Configure your AWS credentials
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#     region_name=os.getenv('AWS_DEFAULT_REGION')
# )

# # Define your S3 bucket name
# BUCKET_NAME = 'indo-fashion-dataset'

# # Load the VGG16 model pre-trained on ImageNet, without the top layers
# print("Loading VGG16 model...")
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# # Add custom top layers to get 384-dimensional features
# x = base_model.output
# x = Flatten()(x)
# x = Dense(1024, activation='relu')(x)
# x = Dropout(0.5)(x)
# feature_layer = Dense(384, activation='relu')(x)
# feature_extraction_model = Model(inputs=base_model.input, outputs=feature_layer)
# feature_extraction_model.summary()

# # Initialize Pinecone
# print("Initializing Pinecone...")
# pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# # Create or connect to the Pinecone index
# index_name = 'clothes'
# if index_name not in pc.list_indexes().names():
#     print(f"Creating index {index_name} with dimension 384...")
#     pc.create_index(
#         name=index_name,
#         dimension=384,  # The dimension size of the feature extraction layer output
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-east-1'
#         )
#     )

# index = pc.Index(index_name)

# class ImageRequest(BaseModel):
#     category: str
#     image: str  # Base64 encoded image

# def preprocess_image(image: Image.Image):
#     image = image.resize((224, 224))
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
#     image = img_to_array(image)
#     image = np.expand_dims(image, axis=0)
#     image = preprocess_input(image)
#     return image

# def extract_features(image: Image.Image):
#     image = preprocess_image(image)
#     features = feature_extraction_model.predict(image).flatten()
#     return features

# @app.post("/get_similar_images/")
# async def get_similar_images_endpoint(request: ImageRequest):
#     print(f"Received request to get similar images for category: {request.category}")
#     uploaded_image = Image.open(BytesIO(base64.b64decode(request.image)))
#     uploaded_image_features = extract_features(uploaded_image)
    
#     # Query Pinecone for the most similar images
#     print("Querying Pinecone for similar images...")
#     query_results = index.query(
#         namespace="ns1",
#         vector=uploaded_image_features.tolist(),
#         top_k=2,
#         include_values=True,
#         include_metadata=True,
#         filter={"category": {"$eq": request.category}}
#     )
    
#     # Fetch the top 2 most similar images
#     top_2_images = [result['id'] for result in query_results['matches']]
#     image_urls = [f'https://{BUCKET_NAME}.s3.amazonaws.com/{img}' for img in top_2_images]
    
#     print(f"Top 2 similar images: {top_2_images}")
#     return {"image_urls": image_urls}

# @app.post("/add_image_to_index/")
# async def add_image_to_index(category: str, image_key: str):
#     print(f"Adding image {image_key} to index under category {category}...")
#     s3_image = get_image_from_s3(image_key)
#     image_features = extract_features(s3_image)
    
#     # Add the image features to Pinecone
#     index.upsert(
#         vectors=[
#             {
#                 "id": image_key,
#                 "values": image_features.tolist(),
#                 "metadata": {"category": category}
#             }
#         ],
#         namespace="ns1"
#     )
#     print(f"Image {image_key} added to index.")
#     return {"status": "success"}

# def get_image_from_s3(key: str):
#     print(f"Fetching image {key} from S3...")
#     obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
#     return Image.open(BytesIO(obj['Body'].read()))


#----------------------------------------------------------------------------------------------------------------------
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import boto3
# import os
# from dotenv import load_dotenv
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from pinecone import Pinecone, ServerlessSpec
# import base64
# from transformers import CLIPProcessor, CLIPModel
# import torch
# import re

# app = FastAPI()

# # Load environment variables from .env file
# load_dotenv()

# # Configure your AWS credentials
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#     region_name=os.getenv('AWS_DEFAULT_REGION')
# )

# # Define your S3 bucket name
# BUCKET_NAME = 'indo-fashion-dataset'

# # Initialize CLIP model and processor
# model_name = "openai/clip-vit-base-patch32"
# clip_model = CLIPModel.from_pretrained(model_name)
# clip_processor = CLIPProcessor.from_pretrained(model_name)

# # Initialize Pinecone
# print("Initializing Pinecone...")
# pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# # Create or connect to the Pinecone index
# index_name = 'clothes2'
# if index_name not in pc.list_indexes().names():
#     print(f"Creating index {index_name} with dimension 512...")
#     pc.create_index(
#         name=index_name,
#         dimension=512,  # The dimension size of the feature extraction layer output
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',
#             region='us-east-1'
#         )
#     )

# index = pc.Index(index_name)

# class ImageRequest(BaseModel):
#     image: str  # Base64 encoded image
#     prompt: str

# def preprocess_image(image: Image.Image):
#     return clip_processor(images=image, return_tensors="pt")["pixel_values"]

# def extract_features(image: Image.Image):
#     inputs = preprocess_image(image)
#     with torch.no_grad():
#         features = clip_model.get_image_features(pixel_values=inputs).flatten().cpu().numpy()
#     return features

# def find_output_category(prompt: str):
#     categories = {
#         'pants': ['pants', 'pant'],
#         'shoes': ['shoes', 'shoe'],
#         'shirt': ['shirt', 'shirts'],
#         'shorts': ['shorts', 'short'],
#         'sneakers': ['sneakers', 'sneaker'],
#         't-shirt': ['tshirt', 'tshirts', 't shirt', 't shirts']
#     }
#     for key, values in categories.items():
#         if any(word in prompt.lower() for word in values):
#             print("hoolallala")
#             print(prompt)
#             print(key)
#             return key
#     raise ValueError("No valid category found in prompt")

# def parse_prompt(prompt: str):
#     category = find_output_category(prompt)
#     num_items = 2  # Default number of items

#     # Extract the number of items
#     num_items_match = re.search(r'\d+', prompt)
#     if num_items_match:
#         num_items = int(num_items_match.group())

#     # Ensure at least 1 unique results
#     num_items = max(1, num_items)

#     return num_items, category

# @app.post("/get_similar_images/")
# async def get_similar_images_endpoint(request: ImageRequest):
#     print(f"Received request to get similar images with prompt: {request.prompt}")
#     uploaded_image = Image.open(BytesIO(base64.b64decode(request.image)))
#     uploaded_image_features = extract_features(uploaded_image)
    
#     try:
#         num_items, category = parse_prompt(request.prompt)
#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))

#     # Query Pinecone for the most similar images
#     print(f"Querying Pinecone for {num_items} items in category: {category}")
#     query_results = index.query(
#         namespace="ns1",
#         vector=uploaded_image_features.tolist(),
#         top_k=num_items,
#         include_values=True,
#         include_metadata=True,
#         filter={"category": {"$eq": category}}
#     )
    
#     # Fetch the most similar images
#     similar_images = [result['id'] for result in query_results['matches']]
#     image_urls = [f'https://{BUCKET_NAME}.s3.amazonaws.com/{img}' for img in similar_images]
    
#     print(f"Similar images: {similar_images}")
#     return {"image_urls": image_urls}

# @app.post("/add_image_to_index/")
# async def add_image_to_index(category: str, image_key: str):
#     print(f"Adding image {image_key} to index under category {category}...")
#     s3_image = get_image_from_s3(image_key)
#     image_features = extract_features(s3_image)
    
#     # Add the image features to Pinecone
#     index.upsert(
#         vectors=[
#             {
#                 "id": image_key,
#                 "values": image_features.tolist(),
#                 "metadata": {"category": category}
#             }
#         ],
#         namespace="ns1"
#     )
#     print(f"Image {image_key} added to index.")
#     return {"status": "success"}

# def get_image_from_s3(key: str):
#     print(f"Fetching image {key} from S3...")
#     obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
#     return Image.open(BytesIO(obj['Body'].read()))

#---------------------------------------------------------------------------------------------------------------------

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import boto3
import os
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from io import BytesIO
from pinecone import Pinecone, ServerlessSpec
import base64
from transformers import CLIPProcessor, CLIPModel
import torch
import re

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

# Initialize CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
clip_processor = CLIPProcessor.from_pretrained(model_name)

# Initialize Pinecone
print("Initializing Pinecone...")
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Create or connect to the Pinecone index
index_name = 'clothes2'
if index_name not in pc.list_indexes().names():
    print(f"Creating index {index_name} with dimension 512...")
    pc.create_index(
        name=index_name,
        dimension=512,  # The dimension size of the feature extraction layer output
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

class ImageRequest(BaseModel):
    image: str  # Base64 encoded image
    prompt: str
    prompt_instructions: str

def preprocess_image(image: Image.Image):
    return clip_processor(images=image, return_tensors="pt")["pixel_values"]

def extract_features(image: Image.Image):
    inputs = preprocess_image(image)
    with torch.no_grad():
        features = clip_model.get_image_features(pixel_values=inputs).flatten().cpu().numpy()
    return features

def find_output_details(prompt: str, prompt_instructions: str):
    categories = {
        'pants': ['pants', 'pant'],
        'shoes': ['shoes', 'shoe'],
        'shirt': ['shirt', 'shirts'],
        'shorts': ['shorts', 'short'],
        'sneakers': ['sneakers', 'sneaker'],
        't-shirt': ['tshirt', 'tshirts', 't shirt', 't shirts']
    }
    colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'purple', 'pink', 'orange', 'brown', 'gray', 'grey']

    number_words = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
        'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
        'nineteen': 19, 'twenty': 20
    }
    category = None
    color = ""
    num_items = 1  # Default number of items

    # Determine the category
    for key, values in categories.items():
        if any(word in prompt.lower() for word in values):
            category = key
            break

    if not category:
        raise ValueError("No valid category found in prompt")

    # Extract the number of items
    num_items_match = re.search(r'\d+', prompt)
    if num_items_match:
        num_items = int(num_items_match.group())
    else:
        # Check for spelled-out numbers
        for word, num in number_words.items():
            if word in prompt.lower():
                num_items = num
                break

    # Ensure at least 1 unique results
    num_items = max(1, num_items)

    # Determine the color
    for col in colors:
        if col in prompt.lower():
            color = col
            break

    return num_items, category, color

@app.post("/get_similar_images/")
async def get_similar_images_endpoint(request: ImageRequest):
    print(f"Received request to get similar images with prompt: {request.prompt} **Instructions:** {request.prompt_instructions}")
    uploaded_image = Image.open(BytesIO(base64.b64decode(request.image)))
    uploaded_image_features = extract_features(uploaded_image)
    
    try:
        num_items, category, color = find_output_details(request.prompt, request.prompt_instructions)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create Pinecone filter
    pinecone_filter = {"category": {"$eq": category}}
    if color:
        pinecone_filter["color"] = {"$eq": color}

    # Query Pinecone for the most similar images
    print(f"Querying Pinecone for {num_items} items in category: {category}, color: {color}")
    query_results = index.query(
        namespace="ns1",
        vector=uploaded_image_features.tolist(),
        top_k=num_items,
        include_values=True,
        include_metadata=True,
        filter=pinecone_filter
    )
    
    # Fetch the most similar images
    similar_images = [result['id'] for result in query_results['matches']]
    image_urls = [f'https://{BUCKET_NAME}.s3.amazonaws.com/{img}' for img in similar_images]
    
    print(f"Similar images: {similar_images}")
    return {"image_urls": image_urls}

@app.post("/add_image_to_index/")
async def add_image_to_index(category: str, color: str, image_key: str):
    print(f"Adding image {image_key} to index under category {category}, color {color}...")
    s3_image = get_image_from_s3(image_key)
    image_features = extract_features(s3_image)
    
    # Add the image features to Pinecone
    index.upsert(
        vectors=[
            {
                "id": image_key,
                "values": image_features.tolist(),
                "metadata": {"category": category, "color": color}
            }
        ],
        namespace="ns1"
    )
    print(f"Image {image_key} added to index.")
    return {"status": "success"}

def get_image_from_s3(key: str):
    print(f"Fetching image {key} from S3...")
    obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=key)
    return Image.open(BytesIO(obj['Body'].read()))

