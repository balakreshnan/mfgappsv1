import streamlit as st

from labelver import labelverfication   
from yoloinf import yoloinf
from imgpdf import pdf_asimage
from yolopdf  import loadpdf
from pdftext import pdftext

# Set page size
st.set_page_config(
    page_title="Gen AI Application Validation",
    page_icon=":rocket:",
    layout="wide",  # or "centered"
    initial_sidebar_state="expanded"  # or "collapsed"
)

# Load your CSS file
def load_css(file_path):
    with open(file_path, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Call the function to load the CSS
load_css("styles.css")

st.logo("bblogo1.png")
st.sidebar.image("bblogo1.png", use_column_width=True)

# Sidebar navigation
nav_option = st.sidebar.selectbox("Navigation", ["Home", 
                                                 "Label Verification", "Yoloinf",
                                                 "Imgpdf", "pdfimage",
                                                 "PdfText",
                                                 "Img3D","About"])

# Display the selected page
if nav_option == "Label Verification":
    labelverfication()
elif nav_option == "Label Verification":
    labelverfication()
elif nav_option == "Imgpdf":    
    pdf_asimage()
elif nav_option == "Yoloinf":
    yoloinf()
elif nav_option == "pdfimage":
    loadpdf()
elif nav_option == "PdfText":
    pdftext()
#elif nav_option == "VisionAgent":
#    vaprocess()

#st.sidebar.image("microsoft-logo-png-transparent-20.png", use_column_width=True)