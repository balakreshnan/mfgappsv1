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
from ultralytics import YOLO


cfg_model_path = 'yolov5s.pt'
model = None
confidence = .25

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

#https://docs.ultralytics.com/usage/python/ - training
#https://docs.ultralytics.com/modes/predict/#plotting-results
#https://docs.ultralytics.com/modes/predict/#boxes

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
    
def yoloinf():
    # global variables
    global model, confidence, cfg_model_path

    count = 0
    temp_file_path = ""
    pdf_bytes = None
    rfpcontent = {}
    rfplist = []
    #tab1, tab2, tab3, tab4 = st.tabs('RFP PDF', 'RFP Research', 'Draft', 'Create Word')
    modeloptions1 = ["gpt-4o-2", "gpt-4o-g", "gpt-4o", "gpt-4-turbo", "gpt-35-turbo"]
    # imgfile = "DunnesStoresImmuneClosedCups450gLabel.jpg"
    docfile = "LabelVerificationblank.docx"
    imgfile = "bus.jpg"

    # device options
    if torch.cuda.is_available():
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=False, index=0)
    else:
        device_option = st.sidebar.radio("Select Device", ['cpu', 'cuda'], disabled=True, index=0)

    # load model
    #model = load_model(cfg_model_path, device_option)

    #model.classes = list(model.names.keys())

    # Create a dropdown menu using selectbox method
    selected_optionmodel1 = st.selectbox("Select an Model:", modeloptions1)
    count += 1

    user_input = st.text_input("Enter the question to ask the AI model", "Compare the image with label specifications")

    image = Image.open(imgfile)
    #img = image

    st.write("Running YOLOv5 inference...")
    #detected_img, results = detect_objects1(image)

    #image = convert_rgb_to_rgba(image)
    print(f"Image mode: {image.mode}")

    # Run inference
    results = model(imgfile)  # return a list of Results objects
    img = None
    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        img = im_rgb
            
    # Display the image with detected objects
    st.image(img, caption='Detected Objects', use_column_width=True)