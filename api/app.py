from fastapi import FastAPI # импорт класса fastapi для создания обьекта приложения
from fastapi.responses import JSONResponse

import uvicorn

import numpy as np

from PIL import Image # обработка изображений

import torch
import torch.nn.functional as F
import numpy as np

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from cnn.create_model import CNN

app = FastAPI()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy"
    }

# эндпойнт для предсказания
@app.post("/predict")
async def predict():
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)