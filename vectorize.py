# import os
# import boto3
# import numpy as np
# from PIL import Image
# from io import BytesIO
# from dotenv import load_dotenv
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras.preprocessing.image import img_to_array
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from pinecone import Pinecone, ServerlessSpec

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

# # Define categories and initialize vectors list
# categories = ['pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt']
# vectors = []

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
#     # Debugging: print feature length and features
#     print(f"Extracted features length: {len(features)}")
#     print(f"Extracted features: {features[:10]}...")  # Print first 10 features for inspection
#     return features

# for category in categories:
#     print(f"Adding images for category: {category}")
#     prefix = f'clothes/train/{category}/'
#     response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
#     image_keys = [obj['Key'] for obj in response.get('Contents', [])]

#     for image_key in image_keys:
#         print(f"Processing image: {image_key}")
#         obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=image_key)
#         image = Image.open(BytesIO(obj['Body'].read()))

#         # Extract features
#         features = extract_features(image)

#         # Add to vectors list
#         vectors.append({
#             "id": image_key,
#             "values": features.tolist(),
#             "metadata": {"category": category}
#         })

# # Upsert vectors into Pinecone
# if vectors:
#     print(f"Upserting {len(vectors)} vectors into Pinecone...")
#     index.upsert(vectors=vectors, namespace="ns1")



#--------------------------------------------------------------------------------------------------
import os
import boto3
import numpy as np
from PIL import Image, ImageColor
from io import BytesIO
from dotenv import load_dotenv
from transformers import CLIPProcessor, CLIPModel
from pinecone import Pinecone, ServerlessSpec
import torch
from skimage.segmentation import slic
from skimage.color import rgb2lab, deltaE_cie76
import warnings

# Ignore warnings from skimage
warnings.filterwarnings("ignore")

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

# Define categories and initialize vectors list
categories = ['pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt']
vectors = []

colors = ['black', 'white', 'red', 'blue', 'green', 'yellow', 'purple', 'pink', 'orange', 'brown', 'gray', 'grey']

def preprocess_image(image: Image.Image):
    return clip_processor(images=image, return_tensors="pt")["pixel_values"]

def extract_features(image: Image.Image):
    inputs = preprocess_image(image)
    with torch.no_grad():
        features = clip_model.get_image_features(pixel_values=inputs).flatten().cpu().numpy()
    return features

def detect_color(image: Image.Image):
    image = image.convert('RGB')
    image_array = np.array(image)
    
    # Perform SLIC segmentation
    segments = slic(image_array, n_segments=50, compactness=10, sigma=1)

    # Convert the image to Lab color space
    lab_image = rgb2lab(image_array)
    background_color = lab_image[0, 0]

    unique_segments, counts = np.unique(segments, return_counts=True)
    if len(counts) > 1:
        largest_segment = unique_segments[np.argsort(counts)[-2]]
    else:
        # If segmentation fails, fallback to using the whole image
        largest_segment = unique_segments[0]

    # Mask the largest segment
    mask = segments == largest_segment
    object_color_lab = np.mean(lab_image[mask], axis=0)

    # Find the closest color from the predefined list
    closest_color = min(colors, key=lambda color: deltaE_cie76(object_color_lab, rgb2lab(np.array(ImageColor.getrgb(color)).reshape(1, 1, 3))[0, 0]))
    return closest_color

for category in categories:
    print(f"Adding images for category: {category}")
    prefix = f'clothes/train/{category}/'
    response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)
    image_keys = [obj['Key'] for obj in response.get('Contents', [])]

    for image_key in image_keys:
        print(f"Processing image: {image_key}")
        obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=image_key)
        image = Image.open(BytesIO(obj['Body'].read()))

        # Extract features
        features = extract_features(image)

        # Detect color
        color = detect_color(image)

        # Add to vectors list
        vectors.append({
            "id": image_key,
            "values": features.tolist(),
            "metadata": {"category": category, "color": color}
        })

# Upsert vectors into Pinecone
if vectors:
    print(f"Upserting {len(vectors)} vectors into Pinecone...")
    index.upsert(vectors=vectors, namespace="ns1")
