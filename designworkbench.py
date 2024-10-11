import os
from openai import AzureOpenAI
import gradio as gr
from dotenv import dotenv_values
import time
from datetime import timedelta
import json
from azure.kusto.data import KustoClient, KustoConnectionStringBuilder, ClientRequestProperties
from azure.kusto.data.exceptions import KustoServiceError
from azure.kusto.data.helpers import dataframe_from_result_table
import streamlit as st
from PIL import Image
import base64
import requests
import io

config = dotenv_values("env.env")

css = """
.container {
    height: 75vh;
}
"""

client = AzureOpenAI(
  azure_endpoint = config["AZURE_OPENAI_ENDPOINT_VISION"], 
  api_key=config["AZURE_OPENAI_KEY_VISION"],  
  api_version="2024-02-01"
  )

# deployment_name = "gpt-4-vision"
#deployment_name = "gpt-4-turbo"
deployment_name = "gpt-4o"

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

def process_image(uploaded_file, selected_optionmodel, user_input, systemprompt):
    returntxt = ""

    if uploaded_file is not None:
        #image = Image.open(os.path.join(os.getcwd(),"temp.jpeg"))
        img_path = os.path.join(os.getcwd(),"temp.jpeg")
        # Open the image using PIL
        # image_bytes = uploaded_file.read()
        #image = Image.open(io.BytesIO(image_bytes))

        base64_image = encode_image(img_path)
        #base64_image = base64.b64encode(uploaded_file).decode('utf-8') #uploaded_image.convert('L')
        imgprompt = f"""
        {systemprompt}

        Question:
        {user_input} 
        """

        # Get the response from the model
        result = processimage(base64_image, imgprompt)

        #returntxt += f"Image uploaded: {uploaded_file.name}\n"
        returntxt = result

    return returntxt

def designworkbench():
    count = 0
    st.title("Upoad your image and ask questions")
    # Split the app layout into two columns
    col1, col2 = st.columns(2)
    with col1:
        selected_optionmodel = ["gpt-4o-2", "gpt-4o-g", "gpt-4o", "gpt-4-turbo", "gpt-35-turbo"]

        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])  
        
        systemprompttxt = """You are a AI Agent. Analyze the image and find details for questions asked.
        Only answer from the data source provided.
        Based on the question asked analyze and provide output what was asked for.
        Provide accurate answers from the image provided."""
        systemprompt = st.text_area("System Prompt", systemprompttxt)
        user_input = st.text_input("Ask a question about the image", "can you organize the design for a business, from post it notes and Split into categories show in the image")

        
    with col2:
        if st.button("Submit"):
            if uploaded_file is not None:
                image_bytes = uploaded_file.read()

                # Open the image using PIL
                image = Image.open(io.BytesIO(image_bytes))   
                st.image(image, caption='Uploaded Image.', use_column_width=True)  
                image.convert('RGB').save('temp.jpeg')
                result = process_image(image, selected_optionmodel, user_input, systemprompt)
                st.markdown(result, unsafe_allow_html=True)
