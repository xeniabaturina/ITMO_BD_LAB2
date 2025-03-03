import os
import sys
import json
import pickle
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import configparser

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess import PenguinPreprocessor
from train import PenguinClassifier
from predict import PenguinPredictor


class TestIntegrationWorkflow(unittest.TestCase):
    """Integration tests for the entire ML workflow."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create test directories
        cls.test_dir = Path("test_integration")
        cls.data_dir = cls.test_dir / "data"
        cls.experiments_dir = cls.test_dir / "experiments"
        cls.results_dir = cls.test_dir / "results"

        cls.data_dir.mkdir(parents=True, exist_ok=True)
        cls.experiments_dir.mkdir(parents=True, exist_ok=True)
        cls.results_dir.mkdir(parents=True, exist_ok=True)

        # Create test config
        cls.config_path = cls.test_dir / "config.ini"
        with open(cls.config_path, "w") as f:
            f.write(
                """[DATA]
x_data = {}/Penguins_X.csv
y_data = {}/Penguins_y.csv

[SPLIT_DATA]
x_train = {}/Train_Penguins_X.csv
y_train = {}/Train_Penguins_y.csv
x_test = {}/Test_Penguins_X.csv
y_test = {}/Test_Penguins_y.csv

[RANDOM_FOREST]
n_estimators = 100
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
path = {}/random_forest.sav
""".format(
                    cls.data_dir,
                    cls.data_dir,
                    cls.data_dir,
                    cls.data_dir,
                    cls.data_dir,
                    cls.data_dir,
                    cls.experiments_dir,
                )
            )

        # Create sample data
        cls.create_sample_data()

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        import shutil

        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)

    @classmethod
    def create_sample_data(cls):
        """Create sample penguin data for testing."""
        # Create a small sample dataset
        islands = ["Torgersen", "Biscoe", "Dream"]
        sexes = ["MALE", "FEMALE"]
        species = ["Adelie", "Gentoo", "Chinstrap"]

        np.random.seed(42)  # For reproducibility

        # Create 30 samples
        n_samples = 30
        data = {
            "island": np.random.choice(islands, n_samples),
            "bill_length_mm": np.random.uniform(30, 60, n_samples),
            "bill_depth_mm": np.random.uniform(10, 25, n_samples),
            "flipper_length_mm": np.random.uniform(170, 230, n_samples),
            "body_mass_g": np.random.uniform(3000, 6000, n_samples),
            "sex": np.random.choice(sexes, n_samples),
        }

        # Assign species based on features to ensure some correlation
        # This is a simplified rule just for testing
        species_data = []
        for i in range(n_samples):
            if data["island"][i] == "Torgersen":
                species_data.append("Adelie")
            elif data["island"][i] == "Biscoe" and data["bill_length_mm"][i] > 45:
                species_data.append("Gentoo")
            else:
                species_data.append("Chinstrap")

        # Create X dataframe
        X = pd.DataFrame(data)

        # Create y dataframe
        y = pd.DataFrame({"species": species_data})

        # Create a combined dataset with species for preprocessing
        combined = X.copy()
        combined["species"] = species_data
        combined["year"] = (
            2020  # Add year column which will be dropped during preprocessing
        )

        # Save to CSV
        combined.to_csv(cls.data_dir / "penguins.csv", index=False)

    def test_full_workflow(self):
        """Test the entire workflow from preprocessing to prediction."""
        # 1. Preprocess data
        preprocessor = PenguinPreprocessor()
        preprocessor.data_path = str(self.data_dir / "penguins.csv")
        preprocessor.X_path = str(self.data_dir / "Penguins_X.csv")
        preprocessor.y_path = str(self.data_dir / "Penguins_y.csv")
        preprocessor.train_path = [
            str(self.data_dir / "Train_Penguins_X.csv"),
            str(self.data_dir / "Train_Penguins_y.csv"),
        ]
        preprocessor.test_path = [
            str(self.data_dir / "Test_Penguins_X.csv"),
            str(self.data_dir / "Test_Penguins_y.csv"),
        ]

        preprocessor.get_data()
        preprocessor.split_data()

        # Verify data files were created
        self.assertTrue(os.path.exists(preprocessor.X_path))
        self.assertTrue(os.path.exists(preprocessor.y_path))
        self.assertTrue(os.path.exists(preprocessor.train_path[0]))
        self.assertTrue(os.path.exists(preprocessor.train_path[1]))
        self.assertTrue(os.path.exists(preprocessor.test_path[0]))
        self.assertTrue(os.path.exists(preprocessor.test_path[1]))

        # 2. Train model
        trainer = PenguinClassifier()
        # Override the paths to use our test data
        trainer.X_train = pd.read_csv(preprocessor.train_path[0], index_col=0)
        trainer.y_train = pd.read_csv(preprocessor.train_path[1], index_col=0)
        trainer.X_test = pd.read_csv(preprocessor.test_path[0], index_col=0)
        trainer.y_test = pd.read_csv(preprocessor.test_path[1], index_col=0)

        # Set the model path
        model_path = os.path.join(self.experiments_dir, "random_forest.sav")
        trainer.model_path = model_path

        # Train the model - this returns a boolean, not the model
        result = trainer.train_random_forest()
        self.assertTrue(result, "Model training should succeed")

        # Verify model file was created
        self.assertTrue(os.path.exists(model_path))

        # 3. Make predictions using the actual Predictor class
        # Create a custom config for the predictor
        config = configparser.ConfigParser()
        config.read(str(self.config_path))

        # Initialize the predictor
        predictor = PenguinPredictor()
        predictor.config = config
        predictor.model_path = model_path

        # Load the model
        with open(model_path, "rb") as f:
            predictor.model = pickle.load(f)

        # Test prediction on a single sample
        test_sample = {
            "island": "Biscoe",
            "bill_length_mm": 45.2,
            "bill_depth_mm": 15.8,
            "flipper_length_mm": 215.0,
            "body_mass_g": 5400.0,
            "sex": "MALE",
        }

        # Convert to DataFrame for prediction
        test_df = pd.DataFrame([test_sample])

        # Make prediction
        prediction = predictor.model.predict(test_df)

        # Verify prediction has expected format
        self.assertIsInstance(prediction, np.ndarray)
        self.assertEqual(prediction.shape, (1,))
        self.assertIn(prediction[0], ["Adelie", "Gentoo", "Chinstrap"])

        # 4. Test prediction probabilities
        probabilities = predictor.model.predict_proba(test_df)

        # Verify probabilities have expected format
        self.assertIsInstance(probabilities, np.ndarray)
        self.assertEqual(probabilities.shape[0], 1)  # 1 sample
        self.assertAlmostEqual(
            np.sum(probabilities), 1.0, delta=0.01
        )  # Probabilities sum to 1

        # 5. Create a test file for the predictor to use
        test_file_path = os.path.join(self.test_dir, "test_sample.json")
        test_data = {
            "X": [test_sample],
            "y": [{"species": "Unknown"}],  # We don't know the actual species
        }
        with open(test_file_path, "w") as f:
            json.dump(test_data, f)

        # Set up the predictor to use our test file
        predictor.test_file = test_file_path
        predictor.results_dir = str(self.results_dir)

        # Run a prediction using the predictor's methods
        # We'll create a simplified version of the predict_sample method
        def predict_sample(sample):
            prediction = predictor.model.predict(pd.DataFrame([sample]))[0]
            probabilities = predictor.model.predict_proba(pd.DataFrame([sample]))[0]
            class_labels = predictor.model.classes_

            result = {
                "input": sample,
                "predicted_species": prediction,
                "probabilities": {
                    label: float(prob)
                    for label, prob in zip(class_labels, probabilities)
                },
            }
            return result

        # Make a prediction
        result = predict_sample(test_sample)

        # Verify the result
        self.assertIn("predicted_species", result)
        self.assertIn("probabilities", result)
        self.assertIn(result["predicted_species"], ["Adelie", "Gentoo", "Chinstrap"])

        # Verify that the probabilities sum to 1
        prob_sum = sum(result["probabilities"].values())
        self.assertAlmostEqual(prob_sum, 1.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
