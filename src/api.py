import configparser
import os
import pickle
import pandas as pd
import traceback
from flask import Flask, request, jsonify

from logger import Logger

SHOW_LOG = True
app = Flask(__name__)
logger = Logger(SHOW_LOG)
log = logger.get_logger(__name__)


class ModelService:
    """
    Class for serving the penguin classification model via an API.
    """

    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config.read("config.ini")
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.model_path = os.path.join(self.project_path, "random_forest.sav")

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            log.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            log.error(f"Error loading model: {e}")
            log.error(traceback.format_exc())
            self.model = None

    def predict(self, data):
        """
        Make predictions with the loaded model.

        Args:
            data (dict): Input data for prediction.

        Returns:
            dict: Prediction results.
        """
        try:
            input_df = pd.DataFrame([data])
            species = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            class_labels = self.model.classes_

            prob_dict = {
                label: float(prob) for label, prob in zip(class_labels, probabilities)
            }

            return {
                "success": True,
                "predicted_species": species,
                "probabilities": prob_dict,
            }

        except Exception as e:
            log.error(f"Error during prediction: {e}")
            log.error(traceback.format_exc())
            return {"success": False, "error": str(e)}


model_service = ModelService()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    if model_service.model is not None:
        return jsonify({"status": "healthy", "model_loaded": True}), 200
    else:
        return jsonify({"status": "unhealthy", "model_loaded": False}), 500


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Expected JSON input:
    {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "male"
    }
    """
    if model_service.model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        required_fields = [
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
        missing_fields = [field for field in required_fields if field not in data]

        if missing_fields:
            return jsonify({"error": f"Missing required fields: {missing_fields}"}), 400

        result = model_service.predict(data)

        if result["success"]:
            return jsonify(result), 200
        else:
            return jsonify(result), 500

    except Exception as e:
        log.error(f"Error in predict endpoint: {e}")
        log.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
