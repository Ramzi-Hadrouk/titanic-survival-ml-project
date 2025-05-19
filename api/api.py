from fastapi import FastAPI
import sys
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
# This allows importing from the src directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.load_model import load_model
from config import config
from typing import List

app = FastAPI(title="Titanic Survival Prediction API")

# Load model at startup
model = load_model()

class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int 
    Fare: float
    Embarked: str

class PredictionResult(BaseModel):
    passenger_id: int
    survival_probability: float
    survived: bool

@app.post("/predict", response_model=List[PredictionResult])
async def predict(passengers: List[PassengerData]):
    """
    Predict survival for Titanic passengers
    
    Example Input:
    [
        {
            "Pclass": 3,
            "Sex": "male",
            "Age": 25,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 7.25,
            "Embarked": "S"
        }
    ]
    """
    # Convert to DataFrame
    df = pd.DataFrame([p.dict() for p in passengers])
    
    # Add derived features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    
    # Encode categoricals
    df["Sex"] = df["Sex"].map(config.SEX_ENCODING)
    df["Embarked"] = df["Embarked"].map(config.EMBARKED_ENCODING)
    
    # Select features
    features = df[config.FEATURES]
    
    # Make predictions
    probabilities = model.predict_proba(features)[:, 1]
    predictions = model.predict(features)
    
    # Format response
    return [
        {
            "passenger_id": idx,
            "survival_probability": float(prob),
            "survived": bool(pred)
        }
        for idx, (prob, pred) in enumerate(zip(probabilities, predictions))
    ]

@app.get("/model_info")
async def get_model_info():
    """Return information about the trained model"""
    return {
        "model_type": type(model).__name__,
        "features_used": config.FEATURES,
        "training_data": str(config.PROCESSED_TRAIN_PATH)
    }