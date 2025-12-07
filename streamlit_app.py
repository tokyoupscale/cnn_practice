import streamlit as st
from streamlit_drawable_canvas import st_canvas

import requests
import json
from io import BytesIO
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="MNIST Recognition",
    page_icon="🔢",
    layout="centered"
)

API_URL = "http://localhost:8000"

st.title("Распознавание рукописных цифр")
st.markdown("""
Веб-приложение для распознавания рукописных цифр с помощью нейронной сети.
            """)

# возвращает RGBA image data в формате 4D numpy array (r, g, b, alpha) on mouse up event как обьект CanvasResult. 
image_data = st_canvas(
     key="canvas", height=200, width=200, drawing_mode="freedraw", display_toolbar=True
)

if image_data is not None:
     st.image(image_data.image_data)
     st.write("предикт: ") # заглушка