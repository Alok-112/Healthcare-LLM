import streamlit as st
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import google.generativeai as genai
import random
import anthropic

dotenv.load_dotenv()

# Models available for Anthropic and Google APIs
anthropic_models = ["claude-3-5-sonnet-20240620"]
google_models = ["gemini-1.5-flash", "gemini-1.5-pro"]

# Convert OpenAI and Streamlit messages to Google Gemini format
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages

# Convert OpenAI and Streamlit messages to Anthropic format
def messages_to_anthropic(messages):
    anthropic_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            anthropic_message = anthropic_messages[-1]
        else:
            anthropic_message = {
                "role": message["role"],
                "content": [],
            }
        if message["content"][0]["type"] == "image_url":
            anthropic_message["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": message["content"][0]["image_url"]["url"].split(";")[0].split(":")[1],
                    "data": message["content"][0]["image_url"]["url"].split(",")[1],
                }
            })
        else:
            anthropic_message["content"].append(message["content"][0])

        if prev_role != message["role"]:
            anthropic_messages.append(anthropic_message)

        prev_role = message["role"]
        
    return anthropic_messages

# Stream responses from LLMs
def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""

    if model_type == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_params["model"],
            generation_config={
                "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
            }
        )
        gemini_messages = messages_to_gemini(st.session_state.messages)

        for chunk in model.generate_content(
            contents=gemini_messages,
            stream=True,
        ):
            chunk_text = chunk.text or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model=model_params["model"] if "model" in model_params else "claude-3-5-sonnet-20240620",
            messages=messages_to_anthropic(st.session_state.messages),
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
        ) as stream:
            for text in stream.text_stream:
                response_message += text
                yield text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})

# Convert image to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()
    return base64.b64encode(img_byte).decode('utf-8')

# Convert file to base64
def file_to_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

# Convert base64 string to image
def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(base64_string)))

# Main Streamlit app
def main():

    # Page configuration
    st.set_page_config(
        page_title="The OmniChat",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # Header
    st.markdown("<h1 style='text-align: center; color: #6ca395;'>🤖 <i>The OmniChat</i> 💬</h1>", unsafe_allow_html=True)

    # Sidebar for API keys and settings
    with st.sidebar:
        cols_keys = st.columns(2)
        with cols_keys[0]:
            default_openai_api_key = os.getenv("API_Key") if os.getenv("API_Key") is not None else ""
            with st.expander("🔐 OpenAI"):
                openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password", key="openai_api_key_input")
        
        with cols_keys[1]:
            default_google_api_key = os.getenv("API_Key") if os.getenv("API_Key") is not None else ""
            with st.expander("🔐 Google"):
                google_api_key = st.text_input("Introduce your Google API Key (https://aistudio.google.com/app/apikey)", value=default_google_api_key, type="password", key="google_api_key_input")

        default_anthropic_api_key = os.getenv("API_Key") if os.getenv("API_Key") is not None else ""
        with st.expander("🔐 Anthropic"):
            anthropic_api_key = st.text_input("Introduce your Anthropic API Key (https://console.anthropic.com/)", value=default_anthropic_api_key, type="password", key="anthropic_api_key_input")

    # Main content
    if (openai_api_key == "" or "sk-" not in openai_api_key) and google_api_key == "" and anthropic_api_key == "":
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")

    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display previous messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])

        # Sidebar for model options and inputs
        with st.sidebar:
            st.divider()
            
            available_models = anthropic_models if anthropic_api_key else []
            available_models += google_models if google_api_key else []
            model = st.selectbox("Select a model:", available_models, index=0)
            model_type = "openai" if model.startswith("gpt") else "google" if model.startswith("gemini") else "anthropic"
            
            model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            audio_response = st.checkbox("Audio response", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state:
                    st.session_state.pop("messages", None)

            st.button("🗑️ Reset conversation", on_click=reset_conversation)

            st.divider()

            # Image/Video Upload
            if model in ["gpt-4-turbo", "gemini-1.5-flash", "gemini-1.5-pro", "claude-3-5-sonnet-20240620"]:
                st.write(f"### **🖼️ Add an image{' or a video' if model_type=='google' else ''}:**")

                def add_image_to_messages():
                    if st.session_state.uploaded_img or ("camera_img" in st.session_state and st.session_state.camera_img):
                        image = Image.open(st.session_state.uploaded_img) if st.session_state.uploaded_img else st.session_state.camera_img
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "data:image/png;base64," + get_image_base64(image),
                                        "alt": "",
                                    }
                                }
                            ]})
                        st.session_state.uploaded_img = None
                        st.session_state.camera_img = None

                def add_video_to_messages():
                    if st.session_state.uploaded_vid:
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "video_file",
                                    "video_file": "data:video/mp4;base64," + file_to_base64(st.session_state.uploaded_vid)
                                }
                            ]})
                        st.session_state.uploaded_vid = None

                st.file_uploader(label="Upload an image", type=["png", "jpg", "jpeg"], key="uploaded_img", on_change=add_image_to_messages)
                st.camera_input(label="Take a picture", key="camera_img", on_change=add_image_to_messages)

                if model_type == "google":
                    st.file_uploader(label="Upload a video", type=["mp4", "webm"], key="uploaded_vid", on_change=add_video_to_messages)

                st.divider()

            # Audio Input
            if model in ["gpt-4-turbo", "gemini-1.5-flash", "gemini-1.5-pro", "claude-3-5-sonnet-20240620"]:
                st.write(f"### **🎤 Add an audio message:**")

                def add_audio_to_messages():
                    if st.session_state.recorded_audio:
                        st.session_state.messages.append({
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio_file",
                                    "audio_file": st.session_state.recorded_audio,
                                }
                            ]})
                        st.session_state.recorded_audio = None

                st.file_uploader(label="Upload an audio", type=["mp3", "wav", "ogg"], key="uploaded_audio", on_change=add_audio_to_messages)
                st.divider()

        # User text input
        if prompt := st.chat_input("Ask anything you want!"):
            st.session_state.messages.append({"role": "user", "content": [{"type": "text", "text": prompt}]})
            with st.chat_message("assistant"):
                if audio_response:
                    for chunk in stream_llm_response(model_params, model_type=model_type, api_key=openai_api_key if model_type == "openai" else google_api_key if model_type == "google" else anthropic_api_key):
                        st.write(chunk)
                else:
                    st.write(prompt)

if __name__ == "__main__":
    main()
