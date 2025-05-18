import pandas as pd
from sklearn.linear_model import LogisticRegression
from typing import Optional
from pathlib import Path
from config import config
from utils.encode_categorical_variables import encode_categorical_variables
from utils.save_models import save_model

def train_titanic_classification_model(train_path: Optional[Path] = None) -> LogisticRegression:

    # Load and preprocess data
    train_data = pd.read_csv(train_path or config.PROCESSED_TRAIN_PATH)

    #Encode categorical variables
    train_data=encode_categorical_variables(train_data)
    # Prepare features and target
    X_train = train_data[config.FEATURES]
    y_train = train_data["Survived"]
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
  
    # Save the trained model
    save_model(model)
    
    return model