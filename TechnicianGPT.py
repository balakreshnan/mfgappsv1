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
import asyncio
import json
import websockets
from pydub import AudioSegment


config = dotenv_values("env.env")

css = """
.container {
    height: 75vh;
}
"""

client = AzureOpenAI(
  azure_endpoint = config["AZURE_OPENAI_ENDPOINT_VISION_4o_LATEST"], 
  api_key=config["AZURE_OPENAI_KEY_VISION_4o_LATEST"],  
  api_version="2024-05-01-preview"
)

model_name = "gpt-4o-2"

search_endpoint = config["AZURE_AI_SEARCH_ENDPOINT"]
search_key = config["AZURE_AI_SEARCH_KEY"]
search_index=config["AZURE_AI_SEARCH_INDEX1"]
SPEECH_KEY = config['SPEECH_KEY']
SPEECH_REGION = config['SPEECH_REGION']
SPEECH_ENDPOINT = config['SPEECH_ENDPOINT']

citationtxt = ""

# Set up your Computer Vision subscription key and endpoint
subscription_key = config["COMPUTER_VISION_KEY"]
endpoint = config["COMPUTER_VISION_ENDPOINT"]

def sumpdfcontent(user_input1, selected_optionmodel1, pdf_bytes, selected_optionsearch):
    returntxt = ""

    rfttext = ""

    if pdf_bytes:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        # Extract and display text from the first page
        if num_pages > 0:
            for page_num in range(num_pages):
                page = reader.pages[page_num]  # Get each page
                text = page.extract_text()  # Extract text from the page
                rfttext += f"### Page {page_num + 1}\n{text}\n\n"  # Accumulate text from each page


    message_text = [
    {"role":"system", "content":f"""You are document AI agent. Be politely, and provide positive tone answers.
     Based on the question do a detail analysis on best answer to provide and give the best answers.
     Here is the content provided:
     {rfttext}


     if the question is outside the bounds of the document context, ask for follow up questions.
     If not sure, ask the user to provide more information."""}, 
    {"role": "user", "content": f"""{user_input1}. Provide summarized content based on the question asked."""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=0.0,
        seed=105,
    )

    returntxt = response.choices[0].message.content
    return returntxt

def load_content():
    returntxt = ""

    rfttext = ""

    pdfoperatorfile = "resources/Operations Manual.pdf"
    pdfoperatorfile1 = "resources/Operation and Maintenance Manual.pdf"
    # Open the PDF file
    with open(pdfoperatorfile, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        # Extract and display text from the first page
        if num_pages > 0:
            for page_num in range(num_pages):
                page = reader.pages[page_num]  # Get each page
                text = page.extract_text()  # Extract text from the page
                rfttext += f"### Page {page_num + 1}\n{text}\n\n"  # Accumulate text from each page
    
    # Open the PDF file
    with open(pdfoperatorfile1, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        # reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        # Extract and display text from the first page
        if num_pages > 0:
            for page_num in range(num_pages):
                page = reader.pages[page_num]  # Get each page
                text = page.extract_text()  # Extract text from the page
                rfttext += f"### Page {page_num + 1}\n{text}\n\n"  # Accumulate text from each page

    returntxt = rfttext
    return returntxt

global pdf_text

def sumpdfcontentfile(user_input1, selected_optionmodel1, pdf_text, selected_optionsearch):
    returntxt = ""

    message_text = [
    {"role":"system", "content":f"""You are document AI agent. Be politely, and provide positive tone answers.
     Based on the question do a detail analysis on best answer to provide and give the best answers.
     Here is the content provided:
     {pdf_text}


     if the question is outside the bounds of the document context, ask for follow up questions.
     If not sure, ask the user to provide more information."""}, 
    {"role": "user", "content": f"""{user_input1}. Provide summarized content based on the question asked."""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=0.0,
        seed=105,
    )

    returntxt = response.choices[0].message.content
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

# Initialize session state for input if not already present
if 'querymfg' not in st.session_state:
    st.session_state.querymfg = 'How to i start the factory line'

# Function to update the session state when the button is clicked
def update_input(query):
    st.session_state.querymfg = query

MODEL = "gpt-4o-realtime-preview"
URL = f"wss://aoaieu1.openai.azure.com/openai/realtime?api-version=2024-10-01-preview&deployment=gpt-4o-realtime-preview'"

def audio_to_item_create_event(audio_bytes: bytes) -> str:
    # Load the audio file from the byte stream
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
    
    # Resample to 24kHz mono pcm16
    pcm_audio = audio.set_frame_rate(24000).set_channels(1).set_sample_width(2).raw_data
    
    # Encode to base64 string
    pcm_base64 = base64.b64encode(pcm_audio).decode()
    
    event = {
        "type": "conversation.item.create", 
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_audio", 
                "audio": encoded_chunk
            }]
        }
    }
    return json.dumps(event)

def techniciangpt():
    count = 0
    temp_file_path = ""
    pdf_bytes = None
    rfpcontent = {}
    rfplist = []
    pdf_text = load_content()


    col1, col2 = st.columns([1, 1])
    with col1:
        modeloptions1 = ["gpt-4o-2", "gpt-4o-g", "gpt-4o", "gpt-4-turbo", "gpt-35-turbo"]
        # Create a dropdown menu using selectbox method
        selected_optionmodel1 = st.selectbox("Select an Model:", modeloptions1)
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

        if st.button("Record Audio"):
            #record_audio()
            querytxt = recognize_from_microphone()
            print('Text recognized:', querytxt)
            # query = querytxt
            update_input(querytxt)

        querymfg = st.text_input("Enter your question", st.session_state.querymfg)

        if pdf_file is not None:
            pdf_bytes = pdf_file.read()
            temp_file_path = f"temp_pdf_{count}.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(pdf_bytes)
            count += 1

        st.image("resources/csifactoryengdraw1.jpg", use_column_width=True)
        pdf_file = "resources/AU-0014367.pdf"
        images = pdf_to_images(pdf_file, zoom=2.0)
        for i, img in enumerate(images):
            st.image(img, caption=f"Page {i+1}", use_column_width=True)
    
    with col2:
        st.write("This is a placeholder for the PDF content")
        if st.button("Process"):
            if pdf_file is not None:
                rttext = sumpdfcontentfile(querymfg, selected_optionmodel1, pdf_text, search_index)
                st.markdown(rttext, unsafe_allow_html=True)
