from train import train_titanic_classification_model
from test import test_titanic_classification_model
from config import config

if __name__ == "__main__":
    try:
        # Train the model
        print("🚂 Training model...")
        model = train_titanic_classification_model()
        print("✅ Model training completed.\n")
        
        # Test the model
        print("🧪 Evaluating model...")
        test_titanic_classification_model(model=model)
        
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")