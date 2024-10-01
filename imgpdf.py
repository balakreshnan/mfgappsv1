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

# Convert the PIL image to base64
def encode_image_to_base64(pil_image):
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_image

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_image_buffer(img):
    return base64.b64encode(img.read()).decode('utf-8')
    
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

def process_image_img(img, selected_optionmodel, user_input):
    returntxt = ""

    if img is not None:
        #image = Image.open(os.path.join(os.getcwd(),"temp.jpeg"))
        # img_path = os.path.join(os.getcwd(), uploaded_file)
        # Open the image using PIL
        #image_bytes = uploaded_file.read()
        #image = Image.open(io.BytesIO(image_bytes))

        base64_image = encode_image_to_base64(img)
        #base64_image = base64.b64encode(uploaded_file).decode('utf-8') #uploaded_image.convert('L')
        imgprompt = f"""You are a image search AI Expert. Based on the question asked Analyze the image and find details.
        Only answer from the data source provided.
        can you extract details of this questions in image and provide.
        respond as json format to say if the search query is found in image or not.
        json format should be  search_query_found and value should be true or false.

        Question:
        {user_input} 
        """

        # Get the response from the model
        result = processimage(base64_image, imgprompt)

        #returntxt += f"Image uploaded: {uploaded_file.name}\n"
        returntxt = result

    return returntxt

def speech_to_text_extract(text, selected_optionmodel1):
    returntxt = ""

    start_time = time.time()

    message_text = [
    {"role":"system", "content":"""You are a Lanugage AI Agent, based on the text provided, extract intent and also the value provided.
     For example change sugar from 5g to 10g. change sugar to 10g.
     Provide the extracted ingredient and value to update only.     
     """}, 
    {"role": "user", "content": f"""Content: {text}."""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=1,
        seed=105,
   )

    returntxt = response.choices[0].message.content + "\n<br>"

    reponse_time = time.time() - start_time 

def recognize_from_microphone():
    returntxt = ""
    # This example requires environment variables named "SPEECH_KEY" and "SPEECH_REGION"
    speech_config = speechsdk.SpeechConfig(subscription=config['SPEECH_KEY'], region=config['SPEECH_REGION'])
    speech_config.speech_recognition_language="en-US"

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    print("Speak into your microphone.")
    speech_recognition_result = speech_recognizer.recognize_once_async().get()

    if speech_recognition_result.reason == speechsdk.ResultReason.RecognizedSpeech:
        print("Recognized: {}".format(speech_recognition_result.text))
        st.write(f"Recognized: {speech_recognition_result.text}")
        speech_to_text_extract(speech_recognition_result.text, "gpt-4o-g")
        returntxt = speech_recognition_result.text
    elif speech_recognition_result.reason == speechsdk.ResultReason.NoMatch:
        print("No speech could be recognized: {}".format(speech_recognition_result.no_match_details))
    elif speech_recognition_result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = speech_recognition_result.cancellation_details
        print("Speech Recognition canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))
            print("Did you set the speech resource key and region values?")
    return returntxt

# Initialize session state for input if not already present
if 'query' not in st.session_state:
    st.session_state.query = 'show me failures for largest abrasive particle'

# Function to update the session state when the button is clicked
def update_input(query):
    st.session_state.query = query

def pdf_asimage():
    count = 0
    temp_file_path = ""
    pdf_bytes = None
    rfpcontent = {}
    rfplist = []
    

    col1, col2 = st.columns([1, 1])

    with col1:

        if st.button("Record Audio"):
            #record_audio()
            querytxt = recognize_from_microphone()
            print('Text recognized:', querytxt)
            # query = querytxt
            update_input(querytxt)
        query = st.text_input("Enter your question", st.session_state.query)

        pdf_file = os.getcwd() + "\\pdf\\1E2500 - slide deck with sample photos of debris types (1).pdf"
        images = pdf_to_images(pdf_file, zoom=2.0)
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_column_width=True)
    with col2:
        if st.button("Search"):
            imgcount = 1
            for i, img in enumerate(images):
                # img.save(f"temp{i}.jpeg")
                #temp_file_path = f"temp{i}.jpeg"
                rfpcontent = process_image_img(img, model_name, query)
                #rfplist.append(rfpcontent)
                print(rfpcontent)
                count += 1
                parsed_data = json.loads(rfpcontent.replace("`", "").replace("json", ""))
                if parsed_data["search_query_found"]:
                    st.image(img, caption=f"Page {i+1}", use_column_width=True)
                    imgcount += 1

                if imgcount == 5:
                    break

            #st.write(rfplist)