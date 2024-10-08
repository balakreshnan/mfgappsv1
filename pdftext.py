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
)

model_name = "gpt-4o-g"

search_endpoint = config["AZURE_AI_SEARCH_ENDPOINT"]
search_key = config["AZURE_AI_SEARCH_KEY"]
search_index=config["AZURE_AI_SEARCH_INDEX1"]
SPEECH_KEY = config['SPEECH_KEY']
SPEECH_REGION = config['SPEECH_REGION']
SPEECH_ENDPOINT = config['SPEECH_ENDPOINT']

citationtxt = ""

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
    st.session_state.query = 'Summarize the content of the PDF'

# Function to update the session state when the button is clicked
def update_input(query):
    st.session_state.query = query

def pdftext():

    col1, col2 = st.columns([1, 1])

    count = 0
    temp_file_path = ""
    pdf_bytes = None
    rfpcontent = {}
    rfplist = []

    with col1:
        st.write("PDF to Text")
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

        query = st.text_input("Enter your question", st.session_state.query)


        if pdf_file is not None:
            pdf_bytes = pdf_file.read()
            temp_file_path = f"temp_pdf_{count}.pdf"
            with open(temp_file_path, "wb") as f:
                f.write(pdf_bytes)
            count += 1
    with col2:
        if st.button("Process"):
            if pdf_file is not None:
                rttext = sumpdfcontent(query, selected_optionmodel1, pdf_bytes, search_index)
                st.markdown(rttext, unsafe_allow_html=True)
