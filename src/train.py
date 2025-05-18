#---------------------------------------------------------------------
# Importing necessary libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_titanic_classification_model(path: str = None) -> LogisticRegression:
    if path:
        # Load the dataset
        train_data = pd.read_csv(path)
    else:
        train_data = pd.read_csv("../data/processed/train_processed.csv")

    # Map Sex: male = 0, female = 1
    train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})

    # Map Embarked: C=0, Q=1, S=2
    train_data["Embarked"] = train_data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare","FamilySize", "IsAlone","Embarked"]
    X_train = train_data[features]
    y_train = train_data["Survived"]
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    return model
