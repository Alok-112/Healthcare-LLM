import streamlit as st
import google.generativeai as genai
import os
import base64
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()

# Configure the Generative AI model
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

# Function to test if the API key is working
def test_api_key():
    try:
        # Attempt to make a minimal request to the API
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content("Test prompt")
        st.success("API key is valid and working.")
        return True
    except Exception as e:
        st.error(f"API key test failed: {e}")
        return False

# Function to encode the image in base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        st.error(f"Error encoding image: {e}")
        return None

# Initialize session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None

def generate_gemini_content(image_path):
    try:
        # Encode the image in base64
        encoded_image = encode_image(image_path)
        if not encoded_image:
            return None

        model = genai.GenerativeModel("gemini-pro")
        # Generate prompt for analyzing the image
        prompt = """
        You are a medical practitioner and an expert in analyzing medical related images working for a very reputed hospital. 
        You will be provided with images and you need to identify the anomalies, any disease or health issues. 
        You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. 
        You only need to respond if the image is related to a human body and health issues. 
        You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

        Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

        Now analyze the image (base64 encoded) and answer the above questions in the same structured manner defined above.
        """
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None

def chat_eli(query):
    eli5_prompt = "You have to explain the below piece of information to a five years old. \n" + query
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(eli5_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error generating ELI5 explanation: {e}")
        return "Unable to generate explanation."

# Streamlit UI
st.title("Medical Help using Multimodal LLM")

with st.expander("About this App"):
    st.write("Upload an image to get an analysis from GPT-4.")

# Test API Key at the start
if test_api_key():
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Temporary file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state['filename'] = tmp_file.name

        st.image(uploaded_file, caption='Uploaded Image')

    if st.button('Analyze Image'):
        if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
            st.session_state['result'] = generate_gemini_content(st.session_state['filename'])
            if st.session_state['result']:
                st.markdown(st.session_state['result'], unsafe_allow_html=True)
            os.unlink(st.session_state['filename'])  # Delete the temp file after processing

    if 'result' in st.session_state and st.session_state['result']:
        st.info("Below you have an option for ELI5 to understand in simpler terms.")
        if st.radio("ELI5 - Explain Like I'm 5", ('No', 'Yes')) == 'Yes':
            simplified_explanation = chat_eli(st.session_state['result'])
            st.markdown(simplified_explanation, unsafe_allow_html=True)
else:
    st.error("Please check your API key configuration.")
