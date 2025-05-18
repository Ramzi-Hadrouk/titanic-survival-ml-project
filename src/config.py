from pathlib import Path

# Configuration settings
class Config:
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    
    # Processed data paths
    PROCESSED_TRAIN_PATH = DATA_DIR / "processed/train_processed.csv"
    
    # Raw data paths
    RAW_TEST_PATH = DATA_DIR / "raw/test.csv"
    GENDER_SUBMISSION_PATH = DATA_DIR / "raw/gender_submission.csv"
    
    # Add to the Config class
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_FILE = RESULTS_DIR / "test_results.txt"

   # Add to Config class
    MODELS_DIR = BASE_DIR / "models"
    MODEL_FILE = MODELS_DIR / "titanic_model.pkl"

    # Model features
    FEATURES = [
        "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
        "FamilySize", "IsAlone", "Embarked"
    ]
    
    # Encodings
    SEX_ENCODING = {"male": 0, "female": 1}
    EMBARKED_ENCODING = {"C": 0, "Q": 1, "S": 2}

config = Config()