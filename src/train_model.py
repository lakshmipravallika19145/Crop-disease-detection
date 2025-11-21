# src/train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_random_forest(csv_path="data/features.csv", model_out="models/rf_model.joblib"):
    # Step 1: Load the dataset
    print(f"📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Step 2: Separate features (X) and labels (y)
    X = df.drop(columns=["label", "image_path"], errors="ignore")
    y = df["label"]

    # Step 3: Handle missing values (if any)
    X = X.fillna(0)

    # Step 4: Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"🧠 Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

    # Step 5: Save the training and testing datasets
    os.makedirs("data", exist_ok=True)

    train_df = X_train.copy()
    train_df["label"] = y_train.values
    train_df.to_csv("data/train_features.csv", index=False)

    test_df = X_test.copy()
    test_df["label"] = y_test.values
    test_df.to_csv("data/test_features.csv", index=False)

    print("💾 Saved training data to: data/train_features.csv")
    print("💾 Saved testing data to: data/test_features.csv")

    # Step 6: Train the Random Forest model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # Step 7: Evaluate model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Model Accuracy: {acc*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    print("\n🧾 Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Step 8: Save trained model
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    joblib.dump(clf, model_out)
    print(f"\n💾 Model saved successfully at: {model_out}")

    return clf, acc

if __name__ == "__main__":
    train_random_forest()
