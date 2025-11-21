# src/predict_image.py

import os
import joblib
import pandas as pd
from extract_features import extract_features

def predict_from_image(image_path, model_path="models/rf_model.joblib"):
    """
    Load the trained model, extract features from a given image, and predict the disease.
    """
    # Step 1: Check image file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: The image file was not found at: {image_path}")
        return None

    # Step 2: Load trained model
    if not os.path.exists(model_path):
        print(f"❌ Error: Trained model file not found at: {model_path}")
        return None

    print(f"🔍 Loading model from {model_path} ...")
    model = joblib.load(model_path)

    try:
        print(f"🖼️ Extracting features from: {image_path}")
        feats = extract_features(image_path)
        df = pd.DataFrame([feats])

        # Step 3: Predict disease
        prediction = model.predict(df)[0]
        print(f"\n✅ Predicted Disease: {prediction}")
        return prediction

    except Exception as e:
        print(f"⚠️ Error processing image: {e}")
        return None


if __name__ == "__main__":
    # ✅ Update this path to any valid image from your dataset
    # Example: Tomato_Late_blight, Potato_Early_blight, etc.
    test_image = "C:\\Users\\klaks\\OneDrive\\Documents\\crop_disease_ml\\data\\PlantVillage\\Tomato_Late_blight\\0a3f65fc-ef1c-4aed-b235-46bae4e5c0e7___GHLB2 Leaf 9065.JPG"

    # If you're unsure of image names, run this command in PowerShell:
    # dir data\plantvillage\Tomato_Late_blight | select -first 5

    predict_from_image(test_image)
