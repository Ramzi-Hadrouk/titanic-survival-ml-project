import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Optional
from pathlib import Path
from config import config
from utils.add_family_features import add_family_features
from utils.encode_categorical_variables import encode_categorical_variables

def test_titanic_classification_model(
    model: LogisticRegression,
    test_path: Optional[Path] = None,
    submission_path: Optional[Path] = None
) -> dict:
    """
    Evaluates a trained Titanic classification model on test data.
    
    Args:
        model: Trained logistic regression model
        test_path: Path to the test CSV file. Defaults to config.RAW_TEST_PATH.
        submission_path: Path to the ground truth. Defaults to config.GENDER_SUBMISSION_PATH.
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load and preprocess test data
    test_data = pd.read_csv(test_path or config.RAW_TEST_PATH)
    test_data =add_family_features(test_data)
    test_data = encode_categorical_variables(test_data)
    # Load ground truth
    result = pd.read_csv(submission_path or config.GENDER_SUBMISSION_PATH)
    
    # Prepare features and handle missing values
    test_data = test_data.dropna(subset=config.FEATURES)
    X_test = test_data[config.FEATURES]
    y_test = result.loc[test_data.index, "Survived"]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    
    # Print results
    print(f"✅ Accuracy: {metrics['accuracy']:.4f}")
    print("\n📄 Classification Report:\n", metrics['classification_report'])
    print("\n🧱 Confusion Matrix:\n", metrics['confusion_matrix'])
    
    return metrics