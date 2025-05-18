import pickle
from pathlib import Path
from typing import Any
from config import config

def load_model(path: Path = None) -> Any:
    """
    Load model from .pkl file
    
    Args:
        path: Path to saved model
        
    Returns:
        Loaded model object
    """
    path = path or config.MODEL_FILE
    with open(path, 'rb') as f:
        return pickle.load(f)