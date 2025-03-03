import configparser
import os
import pickle
import pandas as pd
import traceback
import json
import datetime
from flask import Flask, request, jsonify
from pathlib import Path

from .logger import Logger

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
        # Get the project root directory (assuming src is one level below root)
        self.root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = self.root_dir / "config.ini"
        self.config.read(str(config_path))
        
        self.project_path = str(self.root_dir / "experiments")
        self.model_path = str(Path(self.project_path) / "random_forest.sav")

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)
            log.info(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            log.error(f"Model file not found at {self.model_path}")
            self.model = None
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
            # Validate input data types
            self._validate_input_data(data)

            # Prepare input data
            input_df = pd.DataFrame([data])

            # Make prediction
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

        except ValueError as e:
            log.error(f"Input validation error: {e}")
            return {"success": False, "error": str(e), "error_type": "validation_error"}
        except Exception as e:
            log.error(f"Error during prediction: {e}")
            log.error(traceback.format_exc())
            return {"success": False, "error": str(e), "error_type": "prediction_error"}

    def _validate_input_data(self, data):
        """
        Validate input data types and ranges.

        Args:
            data (dict): Input data for prediction.

        Raises:
            ValueError: If input data is invalid.
        """
        # Check island
        if not isinstance(data.get("island"), str):
            raise ValueError("Island must be a string")

        # Check numeric fields
        numeric_fields = {
            "bill_length_mm": (10.0, 60.0),
            "bill_depth_mm": (10.0, 30.0),
            "flipper_length_mm": (150.0, 250.0),
            "body_mass_g": (2500.0, 6500.0),
        }

        for field, (min_val, max_val) in numeric_fields.items():
            value = data.get(field)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a number")
            if value < min_val or value > max_val:
                raise ValueError(f"{field} must be between {min_val} and {max_val}")

        # Check sex
        if not isinstance(data.get("sex"), str):
            raise ValueError("Sex must be a string")

        # Normalize sex field
        data["sex"] = data["sex"].upper()
        if data["sex"] not in ["MALE", "FEMALE"]:
            raise ValueError("Sex must be either 'MALE' or 'FEMALE'")


def log_request(request_data, response_data, endpoint):
    """Log API requests and responses to a file."""
    # Get the project root directory (assuming src is one level below root)
    root_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = str(root_dir / "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir, f"api_requests_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    )

    log_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
        "status_code": 200 if response_data.get("success", False) else 400,
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


model_service = ModelService()


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    response = {
        "status": "healthy" if model_service.model is not None else "unhealthy",
        "model_loaded": model_service.model is not None,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    status_code = 200 if model_service.model is not None else 500
    return jsonify(response), status_code


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
    request_data = request.get_json()

    # Initial validation
    if model_service.model is None:
        response = {
            "success": False,
            "error": "Model not loaded",
            "error_type": "server_error",
        }
        log_request(request_data, response, "predict")
        return jsonify(response), 500

    try:
        if not request_data:
            response = {
                "success": False,
                "error": "No input data provided",
                "error_type": "validation_error",
            }
            log_request(request_data, response, "predict")
            return jsonify(response), 400

        required_fields = [
            "island",
            "bill_length_mm",
            "bill_depth_mm",
            "flipper_length_mm",
            "body_mass_g",
            "sex",
        ]
        missing_fields = [
            field for field in required_fields if field not in request_data
        ]

        if missing_fields:
            response = {
                "success": False,
                "error": f"Missing required fields: {missing_fields}",
                "error_type": "validation_error",
            }
            log_request(request_data, response, "predict")
            return jsonify(response), 400

        result = model_service.predict(request_data)
        log_request(request_data, result, "predict")

        if result["success"]:
            return jsonify(result), 200
        else:
            error_code = 400 if result.get("error_type") == "validation_error" else 500
            return jsonify(result), error_code

    except Exception as e:
        log.error(f"Error in predict endpoint: {e}")
        log.error(traceback.format_exc())
        response = {"success": False, "error": str(e), "error_type": "server_error"}
        log_request(request_data, response, "predict")
        return jsonify(response), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
