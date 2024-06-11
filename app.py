# import streamlit as st
# import requests
# from PIL import Image
# import numpy as np
# import tensorflow as tf

# # Load your pre-trained model
# model = tf.keras.models.load_model('garment_classification_model.h5')

# # Define the classes
# classes = ['pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt']

# def classify_image(image):
#     image = image.resize((224, 224))  # Assuming your model expects 224x224 images
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     predictions = model.predict(image)
#     class_idx = np.argmax(predictions)
#     return classes[class_idx]

# st.title("Garment Classification and Similar Image Retrieval")

# uploaded_file = st.file_uploader("Upload a garment image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Uploaded Image.', use_column_width=True)
#     st.write("")

#     st.write("Classifying...")
#     category = classify_image(image)
#     st.write(f"Predicted category: {category}")

#     st.write("Fetching similar images from S3...")
#     response = requests.post("http://127.0.0.1:8000/get_similar_images/", json={"category": category})
#     if response.status_code == 200:
#         image_urls = response.json()["image_urls"]
#         for url in image_urls:
#             st.image(url, caption=f"Similar Image: {url.split('/')[-1]}", use_column_width=True)
#     else:
#         st.write("Error fetching similar images.")


#-------------------------------------------------------------
import streamlit as st
import requests
from PIL import Image
import numpy as np
import tensorflow as tf

# Load your pre-trained model
model = tf.keras.models.load_model('garment_classification_model.h5')

# Define the classes
classes = ['pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt']

def classify_image(image):
    image = image.resize((224, 224))  # Assuming your model expects 224x224 images
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    class_idx = np.argmax(predictions)
    return classes[class_idx]

st.title("Garment Classification and Similar Image Retrieval")

uploaded_file = st.file_uploader("Upload a garment image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    st.write("Classifying...")
    category = classify_image(image)
    st.write(f"Predicted category: {category}")

    st.write("Fetching similar images from S3...")
    response = requests.post("http://127.0.0.1:8000/get_similar_images/", json={"category": category})
    if response.status_code == 200:
        image_urls = response.json()["image_urls"]
        print(image_urls)
        for url in image_urls:
            st.image(url, caption=f"Similar Image: {url.split('/')[-1]}", use_column_width=True)
    else:
        st.write("Error fetching similar images.")
        st.write(response.text)

