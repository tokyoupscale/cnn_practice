# api/app.py
import io

from fastapi import FastAPI, File, UploadFile #впадлу дышать
from fastapi.responses import JSONResponse
import uvicorn

import torch
from torchvision import transforms 
from torchvision.transforms import InterpolationMode
from PIL import Image

from cnn.create_model import CNN, device, load_model # импорт модели

# проверка девайса и на винду и на мак крч
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


model = load_model()

mnist_transform = transforms.Compose([
    transforms.Resize(interpolation=InterpolationMode.BICUBIC, size=(28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1.0 - x),
    transforms.Normalize((0.1307,), (0.3081,)),  # стандарт для мнист
])

app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Проверяем, что это вообще картинка
    if not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": f"пришло {file.content_type!r} вместо нужного"}
        )

    file_bytes = await file.read()

    try:
        image = Image.open("localdata/local.png")
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"не удалось прочитать файл как изображение: {str(e)}"}
        )
    
    image = mnist_transform(image).unsqueeze(0).to(device)  # [1, 1, 28, 28]

    with torch.no_grad():
        logits = model(image)
        probabilities = torch.softmax(logits, dim=1)[0]

    # переводим в числа
    probabilities_dict = {str(i): float(probabilities[i].item()) for i in range(10)}
    predicted_digit = torch.argmax(probabilities).item()

    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "pic_width": image.shape[-1],
        "pic_height": image.shape[-2],
        "probabilities": probabilities_dict,
        "predicted": predicted_digit,
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)