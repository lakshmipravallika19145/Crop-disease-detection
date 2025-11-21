# src/app_streamlit.py

import streamlit as st
import joblib
import pandas as pd
from extract_features import extract_features
from PIL import Image
import tempfile
import os

# ---------- Page Configuration ----------
st.set_page_config(page_title="🌿 Crop Disease Detection", page_icon="🌱", layout="centered")

# ---------- Title ----------
st.title("🌿 Crop Disease Detection System")
st.markdown("Detect crop leaf diseases using either an **image upload** or by **manually entering feature values**.")

# ---------- Load Model ----------
@st.cache_resource
def load_model(model_path="models/rf_model.joblib"):
    if not os.path.exists(model_path):
        st.error("❌ Trained model not found! Please train it first using `train_model.py`.")
        return None
    return joblib.load(model_path)

model = load_model()

# ---------- Load feature names from training data ----------
@st.cache_data
def load_feature_names(csv_path="data/features.csv"):
    if not os.path.exists(csv_path):
        st.error("❌ features.csv not found! Please run feature extraction first.")
        return []
    df = pd.read_csv(csv_path)
    # Drop label/image_path to get feature columns only
    feature_cols = [c for c in df.columns if c not in ["label", "image_path"]]
    return feature_cols

feature_names = load_feature_names()

# ---------- Mode Selection ----------
mode = st.radio("Choose Prediction Mode:", ["🖼️ Image Upload", "📊 Manual Feature Input"])

# ---------- Mode 1: Image Upload ----------
if mode == "🖼️ Image Upload":
    uploaded_file = st.file_uploader("📸 Upload a leaf image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())

            # Show image
            st.image(temp_file.name, caption="Uploaded Leaf Image", use_column_width=True)

            # Extract features
            st.info("🔍 Extracting features...")
            features = extract_features(temp_file.name)
            df = pd.DataFrame([features])

            if model is not None:
                st.info("🤖 Predicting disease...")
                prediction = model.predict(df)[0]
                st.success(f"✅ **Predicted Disease:** {prediction}")

                with st.expander("📊 Show Extracted Features"):
                    st.dataframe(df.T.rename(columns={0: "Value"}))
            else:
                st.error("❌ Model not loaded.")

            temp_file.close()
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
    else:
        st.info("👆 Please upload a leaf image to start detection.")

# ---------- Mode 2: Manual Feature Input ----------
else:
    st.subheader("📊 Enter Feature Values Manually")

    if not feature_names:
        st.stop()

    cols = st.columns(2)
    feature_values = {}

    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            val = st.number_input(f"{feature}", min_value=0.0, max_value=3000.0, value=100.0)
            feature_values[feature] = val

    if st.button("🔍 Predict Disease"):
        if model is not None:
            try:
                df = pd.DataFrame([feature_values])  # ensure proper column names and order
                df = df[feature_names]  # reorder exactly as training
                st.info("🤖 Predicting disease based on feature values...")
                prediction = model.predict(df)[0]
                st.success(f"✅ **Predicted Disease:** {prediction}")
                with st.expander("📄 Feature Data Used for Prediction"):
                    st.dataframe(df.T.rename(columns={0: "Value"}))
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")
        else:
            st.error("❌ Model not loaded! Please train the model first.")
