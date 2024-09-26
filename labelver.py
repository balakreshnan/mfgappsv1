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

def getrfptopictorespond(user_input1, selected_optionmodel1, pdf_bytes):
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
    {"role":"system", "content":f"""You are Label verification expert AI Agent. Be politely, and provide positive tone answers.
     Only respond with high level topics and avoid details.
     Here is the compliance text: {rfttext}
     If not sure, ask the user to provide more information."""}, 
    {"role": "user", "content": f"""{user_input1}. Extract the topics to respond back high level bullet point only."""}]

    response = client.chat.completions.create(
        model= selected_optionmodel1, #"gpt-4-turbo", # model = "deployment_name".
        messages=message_text,
        temperature=0.0,
        top_p=0.0,
        seed=105,
    )


    returntxt = response.choices[0].message.content
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
    seed=105,
    )

    #print(response.choices[0].message.content)
    return response.choices[0].message.content

def process_image(uploaded_file, selected_optionmodel, user_input):
    returntxt = ""

    if uploaded_file is not None:
        #image = Image.open(os.path.join(os.getcwd(),"temp.jpeg"))
        img_path = os.path.join(os.getcwd(), uploaded_file)
        # Open the image using PIL

        base64_image = encode_image(img_path)
        #base64_image = base64.b64encode(uploaded_file).decode('utf-8') #uploaded_image.convert('L')
        imgprompt = f"""You are a Label expert Expert Agent. Analyze the image and answer the question.
        Only answer from the data source provided.
        Point out any missing specifications based on the label compliance document.
        Also provide recommendation on any improvements we can do based on the label compliance document.

        Question:
        {user_input} 
        """

        # Get the response from the model
        result = processimage(base64_image, imgprompt)

        #returntxt += f"Image uploaded: {uploaded_file.name}\n"
        returntxt = result

    return returntxt

def extracttextfrompdf(pdf_bytes):
    returntxt = ""

    if pdf_bytes:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        num_pages = len(reader.pages)
        st.write(f"Number of pages in the PDF: {num_pages}")
        # Extract and display text from the first page
        if num_pages > 0:
            page = reader.pages[0]  # Get the first page
            text = page.extract_text()  # Extract text from the page
            returntxt = text

    return returntxt

# Function to read Word document
def read_word_file(docx_file):
    doc = docx.Document(docx_file)
    full_text = []
    for paragraph in doc.paragraphs:
        full_text.append(paragraph.text)
    return '\n'.join(full_text)

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

def label_verify(docx_file, selected_optionmodel, imgfile, user_input="Compare the image with label specifications."):
    returntxt = ""
    doctext = ""

    doctext = read_word_file(docx_file)


    if imgfile is not None:
        #image = Image.open(os.path.join(os.getcwd(),"temp.jpeg"))
        img_path = os.path.join(os.getcwd(), imgfile)
        # Open the image using PIL
        #image_bytes = uploaded_file.read()
        #image = Image.open(io.BytesIO(image_bytes))

        base64_image = encode_image(img_path)
        #base64_image = base64.b64encode(uploaded_file).decode('utf-8') #uploaded_image.convert('L')
        imgprompt = f"""You are a Label verification Expert Agent. Analyze the image and compare with label compliance document info.
        Only answer from the data source provided.
        Image has information about labels that goes in product or in manufacturing plant.
        Point out any missing specifications based on the label compliance document.
        Also provide recommendation on any improvements we can do based on the label compliance document.

        Label Compliance Docuement:
        {doctext}

        Question:
        {user_input} 
        """

        # Get the response from the model
        result = processimage(base64_image, imgprompt)

        #returntxt += f"Image uploaded: {uploaded_file.name}\n"
        returntxt = result

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

#@st.cache_resource  # Cache the model to avoid reloading each time
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)
    return model

# Object detection function
def detect_objects(image):

    st.write("Starting to load model...")
    #logging.info("Starting to load model...")

    #model_path = Path("yolov5s.pt")
    # Load the model
    # model = torch.load(model_path, map_location=torch.device('cpu'))  # Map to CPU if not using CUDA
    # model = load_model()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=True)

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



def labelverfication():
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

    col1, col2 = st.columns([1,1])
    with col1:
        st.image(imgfile, caption=f"Dunnes Stores", use_column_width=True)
    
    with col2:
        labelver = label_verify(docfile, selected_optionmodel1, imgfile, user_input)
        st.markdown(labelver, unsafe_allow_html=True)
        imageinfo = extractobjectsfromimage(imgfile, selected_optionmodel1, user_input)
        # st.json(imageinfo)
        outputjson = json.loads(imageinfo)
        
        # Save JSON data to a file  
        with open('output.json', 'w') as json_file:  
            json.dump(outputjson, json_file, indent=4) 
        image = Image.open(imgfile)
        # Create a draw object
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
        
        # https://onnxruntime.ai/docs/genai/tutorials/phi3-v.html#run-on-cpu
        # Display the image  
        #plt.imshow(image)  
        #plt.axis('off')  
        #plt.show()  
        st.image(image, caption="Image with bounding boxes", use_column_width=True)
        st.write("Image with bounding boxes")
        # Perform detection
        st.write("Running YOLOv5 inference...")
        #detected_img, results = detect_objects(image)
        
        # Display the image with detected objects
        #st.image(detected_img, caption='Detected Objects', use_column_width=True)
        