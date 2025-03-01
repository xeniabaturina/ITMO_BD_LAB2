import os
import sys
import unittest
import json
import pickle
from unittest.mock import patch
import tempfile
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import api
from api import ModelService


class MockModel:
    def predict(self, X):
        return ["Adelie"]

    def predict_proba(self, X):
        return [[0.8, 0.1, 0.1]]

    @property
    def classes_(self):
        return ["Adelie", "Gentoo", "Chinstrap"]


class TestAPI(unittest.TestCase):
    """
    Test cases for the API service.
    """

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.experiments_dir = os.path.join(self.test_dir, "experiments")
        os.makedirs(self.experiments_dir, exist_ok=True)

        self.mock_model = MockModel()

        self.model_path = os.path.join(self.experiments_dir, "random_forest.pkl")
        with open(self.model_path, "wb") as f:
            pickle.dump(self.mock_model, f)

        self.test_input = {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "male",
        }

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.test_dir)

    @patch("api.os.getcwd")
    @patch("api.configparser.ConfigParser.read")
    def test_model_service_init(self, mock_read, mock_getcwd):
        """Test ModelService initialization."""
        mock_getcwd.return_value = self.test_dir

        service = ModelService()
        service.model_path = self.model_path

        with open(self.model_path, "rb") as f:
            service.model = pickle.load(f)

        self.assertIsNotNone(service.model, "Model should be loaded")

        service.model_path = os.path.join(self.experiments_dir, "nonexistent.pkl")
        service.model = None

        service = ModelService()
        service.model_path = os.path.join(self.experiments_dir, "nonexistent.pkl")

        self.assertIsNone(service.model, "Model should be None for nonexistent file")

    def test_predict(self):
        """Test prediction functionality."""
        service = ModelService()
        service.model = self.mock_model

        result = service.predict(self.test_input)

        self.assertTrue(result["success"], "Prediction should succeed")
        self.assertEqual(
            result["predicted_species"], "Adelie", "Should predict Adelie species"
        )
        self.assertIn("probabilities", result, "Result should include probabilities")
        self.assertAlmostEqual(
            result["probabilities"]["Adelie"],
            0.8,
            delta=0.01,
            msg="Adelie probability should be approximately 0.8",
        )

    @patch("api.ModelService")
    def test_health_check_endpoint(self, mock_service_class):
        """Test the health check endpoint."""
        test_app = api.app.test_client()

        api.model_service.model = self.mock_model
        response = test_app.get("/health")
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200, "Health check should return 200 OK")
        self.assertEqual(data["status"], "healthy", "Status should be healthy")
        self.assertTrue(data["model_loaded"], "Model should be reported as loaded")

        api.model_service.model = None
        response = test_app.get("/health")
        data = json.loads(response.data)

        self.assertEqual(
            response.status_code,
            500,
            "Health check should return 500 when model not loaded",
        )
        self.assertEqual(data["status"], "unhealthy", "Status should be unhealthy")
        self.assertFalse(data["model_loaded"], "Model should be reported as not loaded")

    @patch("api.ModelService.predict")
    def test_predict_endpoint(self, mock_predict):
        """Test the prediction endpoint."""
        test_app = api.app.test_client()

        mock_predict.return_value = {
            "success": True,
            "predicted_species": "Adelie",
            "probabilities": {"Adelie": 0.8, "Gentoo": 0.1, "Chinstrap": 0.1},
        }

        api.model_service.model = self.mock_model
        response = test_app.post(
            "/predict", json=self.test_input, content_type="application/json"
        )
        data = json.loads(response.data)

        self.assertEqual(
            response.status_code, 200, "Valid prediction should return 200 OK"
        )
        self.assertTrue(data["success"], "Prediction should succeed")
        self.assertEqual(
            data["predicted_species"], "Adelie", "Should predict Adelie species"
        )

        api.model_service.model = None
        response = test_app.post(
            "/predict", json=self.test_input, content_type="application/json"
        )

        self.assertEqual(
            response.status_code, 500, "Should return 500 when model not loaded"
        )

        api.model_service.model = self.mock_model
        invalid_input = self.test_input.copy()
        del invalid_input["island"]

        response = test_app.post(
            "/predict", json=invalid_input, content_type="application/json"
        )

        self.assertEqual(
            response.status_code, 400, "Should return 400 for missing fields"
        )

        response = test_app.post("/predict", json=None, content_type="application/json")

        self.assertEqual(
            response.status_code,
            500,
            "Should return 500 for empty request in test environment",
        )


if __name__ == "__main__":
    unittest.main()
