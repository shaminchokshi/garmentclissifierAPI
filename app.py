# import streamlit as st
# import requests
# from PIL import Image
# import numpy as np
# import tensorflow as tf
# import base64
# from io import BytesIO

# # Load your pre-trained model
# model = tf.keras.models.load_model('garment_classification_model_384d_50.h5')

# # Define the classes
# classes = ['pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt']

# def classify_image(image):
#     image = image.resize((224, 224))  # Assuming your model expects 224x224 images
#     image = np.array(image) / 255.0
#     image = np.expand_dims(image, axis=0)
#     predictions = model.predict(image)
#     class_idx = np.argmax(predictions)
#     return classes[class_idx]

# def encode_image(image):
#     buffered = BytesIO()
#     image.save(buffered, format="JPEG")
#     return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
#     encoded_image = encode_image(image)
#     response = requests.post("http://127.0.0.1:8000/get_similar_images/", json={"category": category, "image": encoded_image})
   
    
#     if response.status_code == 200:
#         image_urls = response.json()["image_urls"]
#         for url in image_urls:
#             st.image(url, caption=f"Similar Image: {url.split('/')[-1]}", use_column_width=True)
#     else:
#         st.write("Error fetching similar images.")
#         st.write(response.text)

# ------------------------------------------------------------------------------------------

import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

# Define the URL of the backend API
backend_url = "http://127.0.0.1:8000/get_similar_images/"

def main():
    st.title("Fashion Image Similarity Search")

    # Upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    prompt = st.text_input("Enter your prompt:")

    prompt_instructions='''
    -a category will always be one word out of these words 'pants', 'shirt', 'shoes', 'shorts', 'sneakers', 't-shirt'
    - given the prompt, correctly identify the output category. then fetch the most similar results in that category.
    - The prompt will have a number as to how many results it wants give those many results and if it is not specified keep it 2 by default
    - and always return 3 unique results
'''

    if uploaded_file and prompt:
        # Convert the image to base64
        image = Image.open(uploaded_file)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Send the image and prompt to the backend
        data = {"image": img_str, "prompt": prompt, "prompt_instructions": prompt_instructions}
        response = requests.post(backend_url, json=data)

        if response.status_code == 200:
            result = response.json()
            for img_url in result["image_urls"]:
                st.image(img_url, caption="Similar Image")
        else:
            st.error("Error: Unable to fetch similar images.")

if __name__ == "__main__":
    main()



        
