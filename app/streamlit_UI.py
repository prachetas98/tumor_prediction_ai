#Install: pip install streamlit pillow
#Run: streamlit run app.py
import streamlit as st
from PIL import Image
import io

st.title("Image Compute Demo")

uploaded = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])
compute = st.button("Compute")

output = st.empty()

if compute:
    if not uploaded:
        output.text("Please upload an image first.")
    else:
        img = Image.open(io.BytesIO(uploaded.read()))
        w, h = img.size
        output.text_area("Output", value=f"Image loaded.\nName: {uploaded.name}\nSize: {w} x {h}px", height=120)