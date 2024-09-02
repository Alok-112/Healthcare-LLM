from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai

load_dotenv()  # take environment variables from .env.

# Configure the Gemini API
os.getenv("GOOGLE_GEMINI_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

## Function to get response from Gemini model
def get_gemini_response(input, image):
    model = genai.GenerativeModel('gemini-1.5-pro')
    if input:
        response = model.generate_content([input, image])
    else:
        response = model.generate_content(image)
    return response.text

## Function to get simplified explanation
def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five years old. \n" + query
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(eli5_prompt)
    return response.text

## Initialize the Streamlit app
st.set_page_config(page_title="Gemini Image Demo")

col1, col2 = st.columns(2)

# Place the image in the first column
with col1:
    st.image("download-Photoroom.png")

# Place the header in the second column
with col2:
    st.header("Gemini Application")

# Instructions and file upload
input = """You are a medical practitioner and an expert in analyzing medical-related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in a detailed manner. Write all the findings, next steps, recommendations, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

Now analyze the image and answer the above questions in the same structured manner defined above.
"""

# Custom CSS for better UI
st.markdown(

    """
    <style>

    [class="block-container css-12oz5g7 egzxvld2"]{
    background-color : #5F9EA0;
    font-size: 18px; 
    }
    [class="css-zt5igj e16nr0p32"]{

    text-align :left ;
    margin-top :100px;
    color :black;

    }
    </style>



    """,
    unsafe_allow_html=True,
)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
image = ""   
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

# Radio button for response type
response_type = st.radio("Choose the type of response:", ["Detailed", "Simplified"])

submit = st.button("Tell me about the image")

## If the submit button is clicked
if submit:
    if response_type == "Detailed":
        response = get_gemini_response(input, image)
    else:
        # Create a detailed response first
        detailed_response = get_gemini_response(input, image)
        # Simplify the detailed response
        response = chat_eli(detailed_response)
        
    st.subheader("The Response is")
    st.write(response)


