import pandas as pd
def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the Titanic dataset by:
    - Adding family features
    
    Args:
        df: Input DataFrame
        
    Returns:
        Preprocessed DataFrame
    """
    df = df.copy()
    
    # Add family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    
    return df