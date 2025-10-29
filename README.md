# Diabetes Prediction API

## Overview
This project provides a machine learning API for predicting diabetes outcomes.
The API is built with FastAPI and containerized with Docker.

## Project Structure
- `train_model.py` → Train and save the ML model.
- `app/main.py` → FastAPI application exposing prediction endpoints.
- `models/` → Directory to store the trained model.
- `requirements.txt` → Python dependencies.
- `Dockerfile` → Docker configuration.

## Setup & Run

### 1. Train the model
```bash
python train_model.py
```

### 2. Run API locally
```bash
uvicorn app.main:app --reload --port 8000
```

### 3. Build Docker image
```bash
docker build -t diabetes-api .
```

### 4. Run Docker container
```bash
docker run -d -p 8000:8000 diabetes-api
```

## Example Request
```json
POST /predict
{
  "feature1": value1,
  "feature2": value2,
  ...
}
```
