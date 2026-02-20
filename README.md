# Potato Disease Classification (FastAPI + Streamlit + Docker)

This repository contains an end-to-end Potato Leaf Disease Classification project:

- **TensorFlow/Keras** model inference (input size **256×256**)
- **FastAPI** backend API (`/predict`)
- **Streamlit** web UI (upload image → prediction + confidence)
- Fully **Dockerized** for easy local run and deployment-ready structure


## Folder Structure

```
.
├── backend/                 # FastAPI inference service (Dockerized)
├── ui/                      # Streamlit UI (Dockerized)
├── scripts/                 # helper scripts (optional)
├── Potato_classification.ipynb
└── .gitignore
```

> Dataset folders (e.g., `PlantVillage/`) are ignored and are NOT uploaded to GitHub.

---

## Features

- Upload potato leaf image from the UI
- Backend returns:
  - Predicted class
  - Confidence score
  - Probabilities for each class
- Supports models that include `preprocess_input` (example: **ResNetV2**)

---

## Requirements

### Recommended (Docker)
- Docker Desktop installed and running

### Without Docker
- Python 3.10+ (3.11 supported)
- pip + venv

---

## Run Locally (Docker)

### 1) Run Backend (FastAPI)
Open **Terminal 1**:

```bash
cd backend
docker build -t potato-backend .
docker run --rm -p 8080:8080 -e PREPROCESS_MODE=resnetv2 potato-backend
```

Test backend:
```bash
curl http://127.0.0.1:8080/health
```

---

### 2) Run UI (Streamlit)
Open **Terminal 2**:

```bash
cd ui
docker build -t potato-ui .
docker run --rm -p 8501:8080 -e API_URL=http://host.docker.internal:8080 potato-ui
```

Open in browser:
- **http://localhost:8501**

Upload an image → click **Predict** 

---

## Backend API Endpoints

### `GET /health`
Returns service status + model input size.

Example:
```json
{
  "status": "ok",
  "input_size": [256, 256],
  "preprocess_mode": "resnetv2"
}
```

---

### `POST /predict`
Upload an image file and get prediction.

**cURL example:**
```bash
curl -X POST "http://127.0.0.1:8080/predict"   -F "file=@/path/to/image.jpg"
```

**Example response:**
```json
{
  "predicted_class": "Potato___Late_blight",
  "confidence": 0.92,
  "probabilities": {
    "Potato___Early_blight": 0.03,
    "Potato___Late_blight": 0.92,
    "Potato___healthy": 0.05
  }
}
```



## Model Notes

- Input: **256×256 RGB**
- Preprocessing: handled by the model (Lambda `preprocess_input`)
- Backend supports loading using `custom_objects` to resolve `preprocess_input`
- `PREPROCESS_MODE` should match training:
  - `resnetv2` (current setup)


## Training Notebook

- `Potato_classification.ipynb` includes:
  - data loading / augmentation
  - model training (transfer learning)
  - evaluation

> Dataset is not included in this repo. Use your local dataset folder.



## 🌍 Deployment (High Level)

This structure can be deployed on:
- AWS (App Runner / ECS / EC2)
- GCP Cloud Run *(billing required)*

Tip: Keep large datasets out of GitHub and store model artifacts in S3/Drive if needed.


## Author

**Ahmed Ali**  
GitHub: https://github.com/ahmedalikeerio
