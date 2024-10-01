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
# model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model
model = YOLO("yolo11n.pt")

# Set up your Computer Vision subscription key and endpoint
subscription_key = config["COMPUTER_VISION_KEY"]
endpoint = config["COMPUTER_VISION_ENDPOINT"]

def extractobjectsfromimage(imgfile, selected_optionmodel, user_input):
    returntxt = ""

    # Define the API version and URL for object detection
    # analyze_url = endpoint + "vision/v4.0/analyze"
    #analyze_url = endpoint + "/computervision/imageanalysis:analyze?api-version=2024-02-01&features=tags,read,caption,denseCaptions,smartCrops,objects,people"
    analyze_url = endpoint + "/computervision/imageanalysis:analyze?api-version=2023-04-01-preview&features=tags,read,caption,denseCaptions,smartCrops,objects,people"

    # Parameters for the request, specifying that we want to analyze objects
    params = {
        "visualFeatures": "Objects"
    }

    # Path to the local image
    image_path = imgfile

    # Headers for the API request, including the content type for binary data
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/octet-stream'
    }

    # Read the image as binary data
    with open(image_path, "rb") as image_data:
        # Send the POST request to the Azure Computer Vision API
        response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
        #response.raise_for_status()  # Raise exception if the call failed

    # Parse the JSON response
    analysis = response.json()
    returntxt = json.dumps(analysis, indent=5)
    #returntxt = analysis
    #print(returntxt)

    # Extract detected objects from the response
    #if "objects" in analysis:
    #    print("Objects detected in the image:")
    #    for obj in analysis["objects"]:
    #        print(f"Object: {obj['object']}, Confidence: {obj['confidence']:.2f}, "
    #            f"Bounding Box: {obj['rectangle']}")
    #else:
    #    print("No objects detected.")

    return returntxt

def extractobjectsfromimage_image(imgfile, selected_optionmodel, user_input):
    returntxt = ""

    # Define the API version and URL for object detection
    # analyze_url = endpoint + "vision/v4.0/analyze"
    #analyze_url = endpoint + "/computervision/imageanalysis:analyze?api-version=2024-02-01&features=tags,read,caption,denseCaptions,smartCrops,objects,people"
    analyze_url = endpoint + "/computervision/imageanalysis:analyze?api-version=2023-04-01-preview&features=tags,read,caption,denseCaptions,smartCrops,objects,people"

    # Parameters for the request, specifying that we want to analyze objects
    params = {
        "visualFeatures": "Objects"
    }

    # Path to the local image
    image_path = imgfile
    # Convert the image to bytes
    image_byte_array = io.BytesIO()
    imgfile.save(image_byte_array, format='PNG')  # Save as PNG
    image_bytes = image_byte_array.getvalue()

    # Headers for the API request, including the content type for binary data
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/octet-stream'
    }

    # Read the image as binary data
    #with open(image_path, "rb") as image_data:
    #    # Send the POST request to the Azure Computer Vision API
    #    response = requests.post(analyze_url, headers=headers, params=params, data=image_data)
    #    #response.raise_for_status()  # Raise exception if the call failed

    response = requests.post(analyze_url, headers=headers, params=params, data=image_bytes)

    # Parse the JSON response
    analysis = response.json()
    returntxt = json.dumps(analysis, indent=5)
    #returntxt = analysis
    #print(returntxt)

    return returntxt

# Function to draw bounding boxes  
def draw_text_bounding_boxes(draw, words):  
    for word in words:  
        box = word["boundingBox"]  
        content = word["content"]  
        # Draw lines between each point to form a box  
        draw.polygon([(box[0], box[1]), (box[2], box[3]), (box[4], box[5]), (box[6], box[7])], outline="green", width=2)  
        # Put text  
        draw.text((box[0], box[1] - 10), content, fill="green")

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

def loadpdf():

    col1, col2 = st.columns([1, 1])
    with col1:
        st.write("Upload a PDF file")
        pdf_file = st.file_uploader("Upload PDF", type=['pdf'])

        if pdf_file is not None:
            pdf_bytes = pdf_file.read() 
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
            # Embedding PDF using an HTML iframe
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="500" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)    
            st.write("PDF uploaded successfully")
            # Convert to base64

    with col2:
        st.write("Upload a Word file")
        if st.button("Upload PDF"):
            if pdf_file is not None:
                pdf_path = pdf_file.name
                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.getbuffer())               
            
                images = pdf_to_images(pdf_file, zoom=2.0)
                for image in images:
                    # Run inference
                    results = model(image)  # return a list of Results objects
                    img = None
                    # Visualize the results
                    for i, r in enumerate(results):
                        # Plot results image
                        im_bgr = r.plot()  # BGR-order numpy array
                        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
                        img = im_rgb
                            
                    # Display the image with detected objects
                    st.image(img, caption='Detected Objects', use_column_width=True)

                    st.write("Objects detected in the image:")
                    imageinfo = extractobjectsfromimage_image(image, "gpt-4o", "")
                    # st.json(imageinfo)
                    outputjson = json.loads(imageinfo)
                    # print('Output JSON:', outputjson)
                    draw = ImageDraw.Draw(image)

                    # Draw bounding boxes  
                    for item in outputjson['denseCaptionsResult']['values']:  
                        box = item['boundingBox']  
                        text = item['text']  
                        draw.rectangle([box['x'], box['y'], box['x'] + box['w'], box['y'] + box['h']], outline="red", width=2)  
                        draw.text((box['x'], box['y']), text, fill="black")  

                    # Draw text from readResult  
                    #for block in outputjson['readResult']['blocks'][0]:  
                    #    for line in block['lines']:  
                    #        polygon = line['boundingPolygon']  
                    #        text = line['text']  
                    #        x = polygon[0]['x']  
                    #        y = polygon[0]['y']  
                    #        draw.text((x, y), text, fill="blue") 
                    words = outputjson["readResult"]["pages"][0]["words"] 
                    draw_text_bounding_boxes(draw, words)
                    st.image(image, caption="Image with bounding boxes", use_column_width=True)