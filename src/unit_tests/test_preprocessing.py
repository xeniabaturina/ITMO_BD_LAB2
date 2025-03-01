import os
import sys
import unittest
import pandas as pd
import shutil
from unittest.mock import patch, MagicMock

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import DataMaker


class TestPreprocessing(unittest.TestCase):
    """
    Test cases for the preprocessing functionality.
    """

    def setUp(self):
        """Set up test environment."""
        # Create test data directory
        self.test_data_dir = os.path.join(os.getcwd(), "test_data")
        os.makedirs(self.test_data_dir, exist_ok=True)

        # Create sample penguin data
        self.sample_data = pd.DataFrame(
            {
                "species": ["Adelie", "Gentoo", "Chinstrap", "Adelie", "Gentoo"],
                "island": ["Torgersen", "Biscoe", "Dream", "Torgersen", "Biscoe"],
                "bill_length_mm": [39.1, 46.5, 49.3, 38.6, 45.2],
                "bill_depth_mm": [18.7, 15.2, 19.5, 17.2, 14.8],
                "flipper_length_mm": [181, 219, 198, 185, 215],
                "body_mass_g": [3750, 5200, 4400, 3800, 5100],
                "sex": ["male", "female", "male", "female", "male"],
                "year": [2007, 2008, 2009, 2007, 2008],
            }
        )

        # Save sample data to CSV
        self.sample_data_path = os.path.join(self.test_data_dir, "penguins.csv")
        self.sample_data.to_csv(self.sample_data_path, index=False)

        # Initialize DataMaker for testing
        self.data_maker = DataMaker()
        # Override paths for testing
        self.data_maker.project_path = self.test_data_dir
        self.data_maker.data_path = self.sample_data_path
        self.data_maker.X_path = os.path.join(self.test_data_dir, "Penguins_X.csv")
        self.data_maker.y_path = os.path.join(self.test_data_dir, "Penguins_y.csv")
        self.data_maker.train_path = [
            os.path.join(self.test_data_dir, "Train_Penguins_X.csv"),
            os.path.join(self.test_data_dir, "Train_Penguins_y.csv"),
        ]
        self.data_maker.test_path = [
            os.path.join(self.test_data_dir, "Test_Penguins_X.csv"),
            os.path.join(self.test_data_dir, "Test_Penguins_y.csv"),
        ]

    def tearDown(self):
        """Clean up after tests."""
        # Remove test data directory
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_get_data(self):
        """Test get_data functionality."""
        # Run get_data
        result = self.data_maker.get_data()

        # Check result
        self.assertTrue(result, "get_data should return True on success")

        # Check that X and y files were created
        self.assertTrue(
            os.path.isfile(self.data_maker.X_path), "X file should be created"
        )
        self.assertTrue(
            os.path.isfile(self.data_maker.y_path), "y file should be created"
        )

        # Load X and y and check content
        X = pd.read_csv(self.data_maker.X_path, index_col=0)
        y = pd.read_csv(self.data_maker.y_path, index_col=0)

        # X should have features and not species or year
        self.assertNotIn(
            "species", X.columns, "X should not contain the 'species' column"
        )
        self.assertNotIn("year", X.columns, "X should not contain the 'year' column")
        self.assertIn(
            "bill_length_mm", X.columns, "X should contain the 'bill_length_mm' column"
        )

        # y should have only the species column
        self.assertIn("species", y.columns, "y should contain the 'species' column")
        self.assertEqual(y.shape[1], 1, "y should have only one column")

    @patch("preprocess.train_test_split")
    def test_split_data(self, mock_train_test_split):
        """Test split_data functionality."""
        # Create test data
        X = pd.DataFrame(
            {
                "island": [
                    "Torgersen",
                    "Biscoe",
                    "Dream",
                    "Torgersen",
                    "Biscoe",
                    "Dream",
                ],
                "bill_length_mm": [39.1, 46.5, 49.3, 38.6, 45.2, 48.5],
                "bill_depth_mm": [18.7, 15.2, 19.5, 17.2, 14.8, 18.9],
                "flipper_length_mm": [181, 219, 198, 185, 215, 195],
                "body_mass_g": [3750, 5200, 4400, 3800, 5100, 4300],
                "sex": ["male", "female", "male", "female", "male", "female"],
            }
        )

        y = pd.DataFrame(
            {
                "species": [
                    "Adelie",
                    "Gentoo",
                    "Chinstrap",
                    "Adelie",
                    "Gentoo",
                    "Chinstrap",
                ]
            }
        )

        # Mock train_test_split to return predetermined splits
        X_train = X.iloc[:4]
        X_test = X.iloc[4:]
        y_train = y.iloc[:4]
        y_test = y.iloc[4:]

        mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)

        # Mock file operations
        with patch("os.path.isfile", return_value=True):
            with patch("pandas.read_csv", side_effect=[X, y]):
                with patch(
                    "preprocess.DataMaker.save_splitted_data", return_value=None
                ):
                    with patch("builtins.open", MagicMock()):
                        with patch("configparser.ConfigParser.write", MagicMock()):
                            # Run split_data
                            result = self.data_maker.split_data(test_size=0.3)

                            # Check result
                            self.assertTrue(
                                result, "split_data should return True on success"
                            )

                            # Verify train_test_split was called with the right arguments
                            mock_train_test_split.assert_called_once()
                            args, kwargs = mock_train_test_split.call_args
                            self.assertEqual(kwargs["test_size"], 0.3)
                            self.assertEqual(kwargs["random_state"], 42)
                            self.assertTrue(
                                "stratify" in kwargs, "stratify should be in kwargs"
                            )

    def test_get_data_error(self):
        """Test get_data error handling."""
        # Create a DataMaker with an invalid data path
        with patch.object(DataMaker, "__init__", return_value=None):
            data_maker = DataMaker()
            data_maker.data_path = os.path.join(self.test_data_dir, "nonexistent.csv")
            data_maker.X_path = self.data_maker.X_path
            data_maker.y_path = self.data_maker.y_path
            data_maker.log = MagicMock()
            data_maker.config = self.data_maker.config

            # Run get_data - should handle the error
            result = data_maker.get_data()

            # Check result
            self.assertFalse(result, "get_data should return False on error")

    def test_save_splitted_data(self):
        """Test save_splitted_data functionality."""
        # Create test data
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Test path
        test_path = os.path.join(self.test_data_dir, "test_save.csv")

        # Run save_splitted_data
        result = self.data_maker.save_splitted_data(test_df, test_path)

        # Check result
        self.assertTrue(result, "save_splitted_data should return True on success")
        self.assertTrue(os.path.isfile(test_path), "File should be created")

        # Load the saved file and check content
        saved_df = pd.read_csv(test_path, index_col=0)
        self.assertEqual(
            saved_df.shape, test_df.shape, "Saved DataFrame should have the same shape"
        )

    def test_save_splitted_data_error(self):
        """Test save_splitted_data error handling."""
        # Create test data
        test_df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Create an invalid path (directory that doesn't exist)
        invalid_path = os.path.join(
            self.test_data_dir, "nonexistent_dir", "test_save.csv"
        )

        # Mock the log to avoid actual error logging
        self.data_maker.log = MagicMock()

        # Run save_splitted_data with invalid path
        result = self.data_maker.save_splitted_data(test_df, invalid_path)

        # Check result
        self.assertFalse(result, "save_splitted_data should return False on error")


if __name__ == "__main__":
    unittest.main()
