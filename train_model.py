"""
Train and save the Iris Flower Classifier model.
Uses scikit-learn's RandomForestClassifier on the built-in Iris dataset.
"""

import os
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def train_model():
    """Train the Iris classifier and save it to disk."""
    print("=" * 50)
    print("  Iris Flower Classifier — Model Training")
    print("=" * 50)

    # Load dataset
    print("\n[1/4] Loading Iris dataset...")
    iris = load_iris()
    X, y = iris.data, iris.target
    print(f"       Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # Split data
    print("[2/4] Splitting into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"       Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    # Train model
    print("[3/4] Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n       Accuracy: {accuracy * 100:.2f}%")
    print("\n       Classification Report:")
    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    for line in report.split("\n"):
        print(f"       {line}")

    # Save model
    print("[4/4] Saving model...")
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "iris_model.pkl")
    joblib.dump(model, model_path)
    print(f"       Model saved to: {model_path}")
    print("\n" + "=" * 50)
    print("  Training Complete!")
    print("=" * 50)

    return model, accuracy


if __name__ == "__main__":
    train_model()
