"""
Iris Flower Classifier — ML Web Application
Built with Flask and scikit-learn
CI/CD Pipeline Lab — ISE 2
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "iris_model.pkl")


def get_model():
    """Load the pre-trained Iris classifier model."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None


# Species info for display
SPECIES_INFO = {
    "setosa": {
        "name": "Iris Setosa",
        "description": "Known for its small petals and distinctive blue-violet flowers.",
        "color": "#6C5CE7",
    },
    "versicolor": {
        "name": "Iris Versicolor",
        "description": "Also called the Blue Flag, it has medium-sized petals with purple veins.",
        "color": "#00B894",
    },
    "virginica": {
        "name": "Iris Virginica",
        "description": "The largest species with broad petals and vibrant violet-blue flowers.",
        "color": "#E17055",
    },
}


@app.route("/")
def home():
    """Render the home page with the prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle prediction requests from the form or API."""
    try:
        # Get features from form or JSON
        if request.is_json:
            data = request.get_json()
            sepal_length = float(data["sepal_length"])
            sepal_width = float(data["sepal_width"])
            petal_length = float(data["petal_length"])
            petal_width = float(data["petal_width"])
        else:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        model = get_model()
        if model is None:
            return jsonify({"error": "Model not found. Please train the model first."}), 500

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        species_names = ["setosa", "versicolor", "virginica"]
        predicted_species = species_names[prediction]
        confidence = float(probabilities[prediction]) * 100

        result = {
            "species": predicted_species,
            "species_info": SPECIES_INFO[predicted_species],
            "confidence": round(confidence, 2),
            "probabilities": {
                name: round(float(prob) * 100, 2)
                for name, prob in zip(species_names, probabilities)
            },
            "input": {
                "sepal_length": sepal_length,
                "sepal_width": sepal_width,
                "petal_length": petal_length,
                "petal_width": petal_width,
            },
        }

        if request.is_json:
            return jsonify(result)
        return render_template("result.html", result=result)

    except (ValueError, KeyError) as e:
        error_msg = f"Invalid input: {str(e)}"
        if request.is_json:
            return jsonify({"error": error_msg}), 400
        return render_template("index.html", error=error_msg)


@app.route("/health")
def health():
    """Health check endpoint for monitoring and CI/CD."""
    model = get_model()
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "version": "1.0.0",
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
