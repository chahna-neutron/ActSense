import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from machine_learning.feature_extraction import extract_sequence_features

DATA_PATH = "machine_learning/data/pose_sequences.csv"
MODEL_PATH = "machine_learning/models/activity_rf.pkl"

def train():
    df = pd.read_csv(DATA_PATH)
    X_raw = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    print(f"Loaded {len(df)} sequences across classes: {np.unique(y)}")

    # Extract posture + motion features for every 30-frame sequence
    X = np.array([extract_sequence_features(row) for row in X_raw])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    print(f"\nValidation Accuracy: {acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    os.makedirs("machine_learning/models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved trained Random Forest model to {MODEL_PATH}")

if __name__ == "__main__":
    train()