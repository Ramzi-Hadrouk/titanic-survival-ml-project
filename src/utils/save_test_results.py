from datetime import datetime
import os
from pathlib import Path
from typing import Optional
from config import config

def save_test_results(metrics: dict, model, file_path: Path = None) -> None:
    """
    Saves test results in Markdown format by appending to file, 
    enabling model comparisons and easy conversion to other formats.
    
    Args:
        metrics: Dictionary containing evaluation metrics
        model: The trained model object
        file_path: Path to save results (defaults to config.RESULTS_FILE)
    """
    file_path = file_path or config.RESULTS_FILE.with_suffix('.md')  # Force .md extension
    os.makedirs(file_path.parent, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    model_name = model.__class__.__name__
    model_params = str(model).replace('\n', '\n    ')
    
    first_entry = not file_path.exists()
    
    with open(file_path, 'a', encoding='utf-8') as f:
        # Write header only for first model
        if first_entry:
            f.write("# Titanic Model Comparison Report\n\n")
            f.write("> Generated on {}\n\n".format(datetime.now().strftime("%Y-%m-%d")))
            f.write("## Features Used\n```\n")
            f.write("\n".join(config.FEATURES) + "\n```\n\n")
        
        # Model section header
        f.write("---\n\n")
        f.write(f"## Model: {model_name}\n")
        f.write(f"*Evaluated at: {timestamp}*\n\n")
        
        # Model parameters
        f.write("### Parameters\n```python\n")
        f.write(f"{model_params}\n```\n\n")
        
        # Metrics
        f.write("### Evaluation Metrics\n")
        f.write(f"- **Accuracy**: {metrics['accuracy']:.4f}\n\n")
        
        f.write("#### Classification Report\n```\n")
        f.write(metrics['classification_report'] + "\n```\n\n")
        
        f.write("#### Confusion Matrix\n```\n")
        f.write(str(metrics['confusion_matrix']) + "\n```\n\n")
        
        # Footer
        f.write(f"*Model evaluation completed in {timestamp}*\n\n")