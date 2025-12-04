from fastapi import FastAPI # импорт класса fastapi для создания обьекта приложения
import uvicorn

import numpy as np

from PIL import Image # обработка изображений

import torch

app = FastAPI()

# эндпойнт для предсказания
@app.post("/predict")
async def predict():
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)