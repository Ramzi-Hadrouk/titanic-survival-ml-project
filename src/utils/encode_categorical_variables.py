from config import config
import pandas as pd
def encode_categorical_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the Titanic dataset by:
    - Encoding categorical variables    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """

    # Encode categorical variables
    df["Sex"] = df["Sex"].map(config.SEX_ENCODING)
    df["Embarked"] = df["Embarked"].map(config.EMBARKED_ENCODING)
    
    return df