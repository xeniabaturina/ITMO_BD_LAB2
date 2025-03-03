import os
import sys
import unittest
import pandas as pd
import numpy as np
import pickle
import shutil
import configparser
import tempfile
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import PenguinPredictor


class MockArgParser:
    """Mock argument parser for testing."""

    def __init__(self, test_type):
        self.test_type = test_type
        self.args = []

    def parse_args(self):
        class Args:
            pass

        args = Args()
        args.tests = self.test_type
        return args

    def add_argument(self, *args, **kwargs):
        """Mock add_argument method."""
        self.args.append((args, kwargs))
        return self


# Define MockModel outside of the test class to make it picklable
class MockModel:
    def predict(self, X):
        # Always predict Adelie for simplicity
        return np.array(["Adelie"] * len(X))


class TestPrediction(unittest.TestCase):
    """
    Test cases for the prediction functionality.
    """

    def setUp(self):
        """Set up test environment."""
        # Create test directories
        self.test_dir = tempfile.mkdtemp()
        self.test_data_dir = os.path.join(self.test_dir, "data")
        self.test_experiments_dir = os.path.join(self.test_dir, "experiments")
        self.test_results_dir = os.path.join(self.test_dir, "results")

        os.makedirs(self.test_data_dir, exist_ok=True)
        os.makedirs(self.test_experiments_dir, exist_ok=True)
        os.makedirs(self.test_results_dir, exist_ok=True)

        # Create sample test data
        self.X_test = pd.DataFrame(
            {
                "island": ["Torgersen", "Biscoe", "Dream"],
                "bill_length_mm": [39.1, 46.5, 49.3],
                "bill_depth_mm": [18.7, 15.2, 19.5],
                "flipper_length_mm": [181, 219, 198],
                "body_mass_g": [3750, 5200, 4400],
                "sex": ["male", "female", "male"],
            }
        )

        self.y_test = pd.DataFrame({"species": ["Adelie", "Gentoo", "Chinstrap"]})

        # Save sample data to CSV
        self.X_test_path = os.path.join(self.test_data_dir, "Test_Penguins_X.csv")
        self.y_test_path = os.path.join(self.test_data_dir, "Test_Penguins_y.csv")

        self.X_test.to_csv(self.X_test_path, index=True)
        self.y_test.to_csv(self.y_test_path, index=True)

        # Create a mock model instance
        self.mock_model = MockModel()
        self.model_path = os.path.join(self.test_experiments_dir, "random_forest.sav")

        # Save the mock model
        with open(self.model_path, "wb") as f:
            pickle.dump(self.mock_model, f)

        # Create config file
        self.config = configparser.ConfigParser()
        self.config["SPLIT_DATA"] = {
            "X_test": self.X_test_path,
            "y_test": self.y_test_path,
        }

        self.config_path = os.path.join(self.test_dir, "config.ini")
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def tearDown(self):
        """Clean up after tests."""
        # Remove test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @patch("predict.os.getcwd")
    @patch("predict.configparser.ConfigParser.read")
    def test_predictor_initialization(self, mock_read, mock_getcwd):
        """Test Predictor initialization."""
        # Setup mocks
        mock_getcwd.return_value = self.test_dir
        mock_read.return_value = None

        # Create predictor with patched environment
        with patch.object(PenguinPredictor, "__init__", return_value=None):
            predictor = PenguinPredictor()
            predictor.config = self.config
            predictor.X_test = self.X_test
            predictor.y_test = self.y_test
            predictor.model_path = self.model_path
            predictor.log = MagicMock()

            # Test initialization
            self.assertIsNotNone(predictor.config, "Config should be initialized")
            self.assertEqual(
                predictor.model_path,
                self.model_path,
                "Model path should be set correctly",
            )

    @patch("predict.os.getcwd")
    @patch("predict.argparse.ArgumentParser")
    def test_smoke_test_prediction(self, mock_arg_parser, mock_getcwd):
        """Test smoke test prediction."""
        # Setup mocks
        mock_getcwd.return_value = self.test_dir
        mock_parser = MockArgParser("smoke")
        mock_arg_parser.return_value = mock_parser

        # Create predictor with mocked environment
        with patch.object(PenguinPredictor, "__init__", return_value=None):
            predictor = PenguinPredictor()
            predictor.config = self.config
            predictor.X_test = self.X_test
            predictor.y_test = self.y_test
            predictor.model_path = self.model_path
            predictor.log = MagicMock()
            predictor.parser = mock_parser

            # Mock the model loading and file operations
            with patch("predict.pickle.load", return_value=self.mock_model):
                with patch("builtins.open", MagicMock()):
                    with patch("json.dump", MagicMock()):
                        with patch("os.makedirs", MagicMock()):
                            # Run predict method for smoke test
                            result = predictor.predict()

                            # Check result
                            self.assertTrue(
                                result, "Smoke test should return True on success"
                            )

    @patch("predict.os.getcwd")
    @patch("predict.configparser.ConfigParser.read")
    @patch("predict.pd.read_csv")
    @patch("predict.argparse.ArgumentParser")
    def test_predictor_full_initialization(
        self, mock_arg_parser, mock_read_csv, mock_read, mock_getcwd
    ):
        """Test full Predictor initialization."""
        # Setup mocks
        mock_getcwd.return_value = self.test_dir
        mock_read.return_value = None
        mock_read_csv.side_effect = [self.X_test, self.y_test]

        # Create mock argument parser
        mock_parser = MockArgParser("smoke")
        mock_arg_parser.return_value = mock_parser

        # Create mock config
        mock_config = configparser.ConfigParser()
        mock_config["SPLIT_DATA"] = {
            "X_test": self.X_test_path,
            "y_test": self.y_test_path,
        }

        # Patch ConfigParser to return our mock config
        with patch("configparser.ConfigParser", return_value=mock_config):
            # Patch Logger
            with patch("predict.Logger", return_value=MagicMock()):
                # Initialize the predictor
                predictor = PenguinPredictor()

                # Verify the predictor was initialized correctly
                self.assertEqual(predictor.X_test.shape, self.X_test.shape)
                self.assertEqual(predictor.y_test.shape, self.y_test.shape)
                self.assertTrue(predictor.model_path.endswith("random_forest.sav"))
                self.assertIsNotNone(predictor.parser)

    @patch("predict.os.getcwd")
    @patch("predict.argparse.ArgumentParser")
    def test_functional_test_prediction(self, mock_arg_parser, mock_getcwd):
        """Test functional test prediction."""
        # Setup mocks
        mock_getcwd.return_value = self.test_dir
        mock_parser = MockArgParser("func")
        mock_arg_parser.return_value = mock_parser

        # Create predictor with mocked environment
        with patch.object(PenguinPredictor, "__init__", return_value=None):
            predictor = PenguinPredictor()
            predictor.config = self.config
            predictor.X_test = self.X_test
            predictor.y_test = self.y_test
            predictor.model_path = self.model_path
            predictor.log = MagicMock()
            predictor.parser = mock_parser

            # Mock the model loading and file operations
            with patch("predict.pickle.load", return_value=self.mock_model):
                with patch("builtins.open", MagicMock()):
                    with patch("json.dump", MagicMock()):
                        with patch("os.makedirs", MagicMock()):
                            # Run predict method for functional test
                            result = predictor.predict()

                            # Check result
                            self.assertTrue(
                                result, "Functional test should return True on success"
                            )

    @patch("predict.os.getcwd")
    @patch("predict.argparse.ArgumentParser")
    def test_model_file_not_found(self, mock_arg_parser, mock_getcwd):
        """Test prediction when model file is not found."""
        # Setup mocks
        mock_getcwd.return_value = self.test_dir
        mock_parser = MockArgParser("smoke")
        mock_arg_parser.return_value = mock_parser

        # Create predictor with mocked environment
        with patch.object(PenguinPredictor, "__init__", return_value=None):
            predictor = PenguinPredictor()
            predictor.config = self.config
            predictor.X_test = self.X_test
            predictor.y_test = self.y_test
            predictor.model_path = os.path.join(
                self.test_experiments_dir, "nonexistent.sav"
            )
            predictor.log = MagicMock()
            predictor.parser = mock_parser

            # Run predict method with nonexistent model file
            result = predictor.predict()

            # Check result
            self.assertFalse(
                result, "Predict should return False when model file is not found"
            )


if __name__ == "__main__":
    unittest.main()
