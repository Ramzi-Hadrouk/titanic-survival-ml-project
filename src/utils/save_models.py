import pickle
from pathlib import Path
from typing import Any
from config import config

def save_model(model: Any, path: Path = None) -> Path:
    """
    Save trained model as .pkl file
    
    Args:
        model: Trained model object
        path: Optional custom path to save model
        
    Returns:
        Path where model was saved
    """
    path = path or config.MODEL_FILE
    path.parent.mkdir(exist_ok=True)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    return path