import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import mlflow

# Initialize MLflow
mlflow.set_tracking_uri("https://991c-35-231-224-155.ngrok-free.app/")  # Set the path to your MLflow tracking server

# Load the PyTorch model from MLflow
MODEL_NAME = 'mercon'
current_stage = 'Production'
model_uri=f"models:/{MODEL_NAME}/{current_stage}"

model = mlflow.pytorch.load_model(model_uri)

# Define data transformations for inference
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define class labels
class_labels = ['Cat', 'Dog']

# Streamlit app
st.title("Cat vs. Dog Classifier")

current_stage = st.sidebar.selectbox("Stage", ("None","Staging", "Production", "Archived"))
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction when a user uploads an image
    if st.button("Classify"):
        image = Image.open(uploaded_image)
        image = data_transform(image).unsqueeze(0)  # Preprocess the image

        # Use the model for inference
        with torch.no_grad():
            model.eval()
            prediction = model(image)

        predicted_class = class_labels[prediction.argmax()]
        st.write(f"Prediction: {predicted_class}")
