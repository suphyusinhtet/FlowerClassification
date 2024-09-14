import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import gdown
import os

# Function to download a model from Google Drive
def download_model_from_drive(drive_url, output_path):
    gdown.download(drive_url, output_path, quiet=False)

# Model Google Drive URLs
my_trained_model_url = 'https://drive.google.com/uc?id=1oxKLETDfVePlCXpUpsXx9pcXrO5Y0UZ7'
mobilenetv2_model_url = 'https://drive.google.com/uc?id=1iC3BskUSOaYYLwyLGqdGg7uPpfrlmp92'
xception_model_url = 'https://drive.google.com/uc?id=1LVkZH_l-VFj7I9BEeqEVj9AHqQfDeOt6'

# Model local paths
my_trained_model_path = 'my_trained_model.h5'
mobilenetv2_model_path = 'mobilenetv2_model.h5'
xception_model_path = 'xception_model.h5'

# Function to load models with caching
@st.cache_resource
def load_cached_model(url, path):
    if not os.path.exists(path):
        download_model_from_drive(url, path)
    return load_model(path)

# Load the models
my_trained_model = load_cached_model(my_trained_model_url, my_trained_model_path)
MobileNetV2_trained_model = load_cached_model(mobilenetv2_model_url, mobilenetv2_model_path)
Xception_trained_model = load_cached_model(xception_model_url, xception_model_path)

# Streamlit UI for uploading an image
st.title("Flower Classification with CNN and Transfer Learning")
uploaded_file = st.file_uploader("Choose an image of a flower...", type="jpg")

if uploaded_file is not None:
    # Load and preprocess the image
    img = image.load_img(uploaded_file, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Scale pixel values to [0, 1]

    # Make predictions
    prediction_with_my_trained_model = my_trained_model.predict(img_array)
    prediction_with_MobileNetV2_trained_model = MobileNetV2_trained_model.predict(img_array)
    prediction_with_Xception_trained_model = Xception_trained_model.predict(img_array)

    # Get the predicted class index
    predicted_class_1 = np.argmax(prediction_with_my_trained_model, axis=-1)
    predicted_class_2 = np.argmax(prediction_with_MobileNetV2_trained_model, axis=-1)
    predicted_class_3 = np.argmax(prediction_with_Xception_trained_model, axis=-1)

    # List of class names
    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    # Display the predictions
    st.write(f"**The predicted class with my trained model is:** {class_names[predicted_class_1[0]]}")
    st.write(f"**The predicted class with MobileNetV2 model is:** {class_names[predicted_class_2[0]]}")
    st.write(f"**The predicted class with Xception model is:** {class_names[predicted_class_3[0]]}")
