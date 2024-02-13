from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
import uvicorn
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi.staticfiles import StaticFiles

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="templates"), name="Static")

Model = load_model("Cnn_model")
class_name = ["Healthy", "Powdery", "Rust"]

@app.get("/ping")
async def ping():
    return "hello, i am alive"

def read_file_as_image(data) -> np.array:
    image = np.array(Image.open(BytesIO(data)))
    image = Image.fromarray(image).resize((256, 256))  # Resize the image
    return np.array(image)

@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    predictions = Model.predict(image_batch)
    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    prediction_result = {"class": predicted_class, "confidence": float(confidence)}
    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction_result})

@app.get("/")
async def upload_file(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == '__main__':
    uvicorn.run(app, host='localhost', port=8080)
