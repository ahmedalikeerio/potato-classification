import os
import json
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

# --- IMPORTANT: add these imports ---
from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenetv2_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

MODEL_PATH = "deploy/potato_model.h5"
CLASS_PATH = "deploy/class_names.json"

# choose which preprocess was used when training
# resnetv2 | mobilenetv2 | efficientnet | none
PREPROCESS_MODE = os.getenv("PREPROCESS_MODE", "resnetv2")

DEFAULT_H, DEFAULT_W = 256, 256

app = FastAPI(title="Potato Disease Classifier API")

model = None
class_names = None
TARGET_H, TARGET_W = DEFAULT_H, DEFAULT_W


from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnetv2_preprocess

def preprocess_input(x):
    return resnetv2_preprocess(x)

def load_keras_model(path: str):
    return tf.keras.models.load_model(
        path,
        compile=False,
        custom_objects={"preprocess_input": preprocess_input}
    )


@app.on_event("startup")
def load_assets():
    global model, class_names, TARGET_H, TARGET_W

    model = load_keras_model(MODEL_PATH)

    # Infer input size if possible
    ishape = model.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    if ishape and len(ishape) >= 3 and ishape[1] and ishape[2]:
        TARGET_H, TARGET_W = int(ishape[1]), int(ishape[2])

    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)


@app.get("/health")
def health():
    return {"status": "ok", "input_size": [TARGET_H, TARGET_W], "preprocess_mode": PREPROCESS_MODE}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()

    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)  # uint8
    img = tf.image.resize(img, (TARGET_H, TARGET_W))
    img = tf.cast(img, tf.float32)  # 0..255 (good)

    x = tf.expand_dims(img, axis=0)

    probs = model.predict(x, verbose=0)[0]
    idx = int(np.argmax(probs))

    return JSONResponse({
        "predicted_class": class_names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    })
