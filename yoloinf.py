import os
from openai import AzureOpenAI
import gradio as gr
from dotenv import dotenv_values
import time
from datetime import timedelta
import json
import streamlit as st
from PIL import Image
import base64
import requests
import io
import autogen
from typing import Optional
from typing_extensions import Annotated
from streamlit import session_state as state
import azure.cognitiveservices.speech as speechsdk
from audiorecorder import audiorecorder
import pyaudio
import wave
import PyPDF2
import docx
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import torch
from pathlib import Path
import numpy as np



config = dotenv_values("env.env")

css = """
.container {
    height: 75vh;
}
"""

client = AzureOpenAI(
  azure_endpoint = config["AZURE_OPENAI_ENDPOINT_VISION"], 
  api_key=config["AZURE_OPENAI_KEY_VISION"],  
  api_version="2024-05-01-preview"
  #api_version="2024-02-01"
  #api_version="2023-12-01-preview"
  #api_version="2023-09-01-preview"
)

#model_name = "gpt-4-turbo"
#model_name = "gpt-35-turbo-16k"
model_name = "gpt-4o-g"

search_endpoint = config["AZURE_AI_SEARCH_ENDPOINT"]
search_key = config["AZURE_AI_SEARCH_KEY"]
search_index=config["AZURE_AI_SEARCH_INDEX1"]
SPEECH_KEY = config['SPEECH_KEY']
SPEECH_REGION = config['SPEECH_REGION']
SPEECH_ENDPOINT = config['SPEECH_ENDPOINT']

citationtxt = ""

#@st.cache_resource  # Cache the model to avoid reloading each time
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    return model

# Object detection function
def detect_objects(image):

    st.write("Starting to load model...")
    #logging.info("Starting to load model...")
    MODEL_PATH = os.getcwd() + "\\yolov5s.pt"

    #model_path = Path("yolov5s.pt")
    # Load the model
    # model = torch.load(model_path, map_location=torch.device('cpu'))  # Map to CPU if not using CUDA
    # model = load_model()
    # 
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

    # If necessary, extract the model from the loaded dictionary (depends on how the model was saved)
    #if isinstance(model, dict) and 'model' in model:
    #    model = model['model']

    

    # Set the model to evaluation mode
    # model.eval()
    st.write("Model loaded successfully!")
    # Convert PIL image to a format YOLOv5 accepts
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Perform inference using the YOLOv5 model
    results = model(img)
    
    # Extract results and draw bounding boxes on the image
    results.render()
    
    # Convert the result back to RGB for displaying in Streamlit
    detected_img = Image.fromarray(results.ims[0])
    
    return detected_img, results.pandas().xyxy[0]  # Image with boxes and DataFrame with results

# Function to load the YOLOv5 model from a .pt file
def load_model1(model_path, device):
    # Load the checkpoint or model file
    checkpoint = torch.load(model_path, map_location=device)

    # If it's a state_dict (OrderedDict), rebuild the YOLOv5 model architecture
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        model_config = checkpoint['model'].yaml  # YOLOv5 architecture (config) is stored in 'model'.yaml
        model = Model(model_config).to(device)  # Create model from config
        model.load_state_dict(checkpoint['model'].state_dict())  # Load model weights
    else:
        model = checkpoint  # Load the model directly if it's a model object

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to perform object detection
def detect_objects1(image):

    # MODEL_PATH = "path/to/yolov5s.pt"  # Update this with the actual path to your model
    MODEL_PATH = os.getcwd() + "\\yolov5s.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise CPU

    # Load the YOLOv5 model
    # model = torch.load(MODEL_PATH, map_location=device)['model'].to(device)
    # model.eval()  # Set the model to evaluation mode
    # Load the model
    model = load_model1(MODEL_PATH, device)
    
    # Convert PIL image to a numpy array
    img = np.array(image)
    
    # Convert the numpy array to a PyTorch tensor and add batch dimension
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0  # Normalize to [0, 1]
    
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Run YOLOv5 model inference
    with torch.no_grad():
        results = model(img)[0]

    # Post-process the results to apply NMS (Non-Maximum Suppression)
    # This step is crucial to filter out overlapping bounding boxes
    from torchvision.ops import nms
    conf_threshold = 0.25
    iou_threshold = 0.45
    boxes = results[:, :4]
    scores = results[:, 4]
    labels = results[:, 5]
    
    # Apply NMS
    indices = nms(boxes, scores, iou_threshold)
    results = results[indices]

    # Convert the result image back to PIL for displaying in Streamlit
    detected_img = Image.fromarray(np.array(image))
    
    return detected_img, results  # Detected image and results

def yoloinf():
    count = 0
    temp_file_path = ""
    pdf_bytes = None
    rfpcontent = {}
    rfplist = []
    #tab1, tab2, tab3, tab4 = st.tabs('RFP PDF', 'RFP Research', 'Draft', 'Create Word')
    modeloptions1 = ["gpt-4o-2", "gpt-4o-g", "gpt-4o", "gpt-4-turbo", "gpt-35-turbo"]
    imgfile = "DunnesStoresImmuneClosedCups450gLabel.jpg"
    docfile = "LabelVerificationblank.docx"



    # Create a dropdown menu using selectbox method
    selected_optionmodel1 = st.selectbox("Select an Model:", modeloptions1)
    count += 1

    user_input = st.text_input("Enter the question to ask the AI model", "Compare the image with label specifications")

    image = Image.open(imgfile)

    st.write("Running YOLOv5 inference...")
    detected_img, results = detect_objects1(image)
        
    # Display the image with detected objects
    st.image(detected_img, caption='Detected Objects', use_column_width=True)