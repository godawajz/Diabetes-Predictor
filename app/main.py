from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import os


class PatientData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float

    #Serum measurements
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

    class Config:
        schema_extra={
            "example": {
                "age": 0.05,
                "sex": 0.05,
                "bmi": 0.06,
                "bp": 0.02,

                "s1": -0.04,
                "s2": -0.04,
                "s3": -0.02,
                "s4": -0.01,
                "s5": 0.01,
                "s6": 0.02
            }
        }



# ========================== FastAPI initialization ==========================
# Initialize FastAPI app
app = FastAPI(
    title="Diabetes Progression Predictor",
    description="Predicts diabetes progression score from physiological features",
    version="1.0.0" )

model_path = os.path.join("models", "diabetes_model.pkl")
with open(model_path, 'rb') as f:
    model = pickle.load(f)


@app.post("/predict")
def predict_progression(patient: PatientData):
# Returns diabetes regression score prediction based on inputs

    features = np.array([[  # prepare input data
        patient.age, patient.sex, patient.bmi, patient.bp,
        patient.s1, patient.s2, patient.s3, patient.s4,
        patient.s5, patient.s6
    ]])
    # ==================== training ====================
    prediction = model.predict(features)[0]
    return {
        "predicted_progression_score": round(prediction, 2),
        "interpretation": get_interpretation(prediction)
    }


def get_interpretation(score):
# Conversion of scores into simple, readable information
    if score < 100:
        return "Below average progression"
    elif score < 150:
        return "Average progression"
    else:
        return "Above average progression"

@app.get("/")
def health_check():
    return {"status": "healthy", "model": "diabetes_progression_v1"}