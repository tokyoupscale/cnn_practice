import streamlit as st
from streamlit_drawable_canvas import st_canvas

import requests
import json
import numpy as np
from PIL import Image

from io import BytesIO

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

# возвращает RGBA image data в формате 4D numpy array (r, g, b, alpha) после mouse up event как обьект CanvasResult. 
canvas_result = st_canvas(
     key="canvas", 
     background_color="#ffffff", 
     height=64, 
     width=64, 
     stroke_width=2, 
     display_toolbar=True
)

if canvas_result is not None:
     # st.write(image_data.json_data)
     img_array = canvas_result.image_data
     img = Image.fromarray(img_array, mode="RGBA")
     
     # buf = BytesIO()
     # img.save(buf, format="PNG")
     # buf.seek(0)
     Image.fromarray(img_array).save("localdata/local.png")
     buf = open("localdata/local.png", "rb").read()

     files = {"file": ("localdata/local.png", buf, "image/png")}

     response = requests.post(f"{API_URL}/predict", files=files)

     st.write("предикт:")
     st.json(response.json())