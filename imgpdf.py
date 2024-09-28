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

# Function to convert PDF pages into images
def pdf_to_images(pdf_path, zoom=2.0):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    images = []

    # Iterate through all the pages
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # Load page
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))  # Render the page as an image
        image = Image.open(io.BytesIO(pix.tobytes("png")))  # Convert to PIL Image
        images.append(image)

    return images

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def processimage(base64_image, imgprompt):
    response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
        "role": "user",
        "content": [
            {"type": "text", "text": f"{imgprompt}"},
            {
            "type": "image_url",
            "image_url": {
                "url" : f"data:image/jpeg;base64,{base64_image}",
            },
            },
        ],
        }
    ],
    max_tokens=2000,
    temperature=0,
    top_p=1,
    )

    #print(response.choices[0].message.content)
    return response.choices[0].message.content

def process_image(uploaded_file, selected_optionmodel, user_input):
    returntxt = ""

    if uploaded_file is not None:
        #image = Image.open(os.path.join(os.getcwd(),"temp.jpeg"))
        img_path = os.path.join(os.getcwd(), uploaded_file)
        # Open the image using PIL
        #image_bytes = uploaded_file.read()
        #image = Image.open(io.BytesIO(image_bytes))

        base64_image = encode_image(img_path)
        #base64_image = base64.b64encode(uploaded_file).decode('utf-8') #uploaded_image.convert('L')
        imgprompt = f"""You are a Constructon drawing AutoCad Expert Agent. Analyze the image and find details for questions asked.
        Only answer from the data source provided.
        Image has information about drawingsprovided.
        can you extract details of this drawings.

        Question:
        {user_input} 
        """

        # Get the response from the model
        result = processimage(base64_image, imgprompt)

        #returntxt += f"Image uploaded: {uploaded_file.name}\n"
        returntxt = result

    return returntxt


def pdf_asimage():
    count = 0
    temp_file_path = ""
    pdf_bytes = None
    rfpcontent = {}
    rfplist = []
    

    col1, col2 = st.columns([1, 1])

    with col1:
        pdf_file = os.getcwd() + "\\pdf"
        images = pdf_to_images(pdf_file, zoom=2.0)
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_column_width=True)