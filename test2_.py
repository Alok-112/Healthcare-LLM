import streamlit as st
from PIL import Image
import io

def main():
    st.title("Photo Upload App")
    st.write("Upload a photo to see it displayed here.")

    # Upload file
    uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # To read file as bytes:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Photo", use_column_width=True)

if __name__ == "__main__":
pip install  streamlit

    main()
