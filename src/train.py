import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import traceback

from logger import Logger

SHOW_LOG = True


class PenguinClassifier:
    """
    Class for training a Random Forest classifier on the penguin dataset.
    """

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")

        self.X_train = pd.read_csv(self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(self.config["SPLIT_DATA"]["y_test"], index_col=0)

        self.project_path = os.path.join(os.getcwd(), "experiments")
        os.makedirs(self.project_path, exist_ok=True)
        self.model_path = os.path.join(self.project_path, "random_forest.sav")

        self.log.info("PenguinClassifier is ready")

    def train_random_forest(self, predict=False) -> bool:
        """
        Train a Random Forest classifier on the penguin dataset.

        Args:
            predict (bool): Whether to make predictions after training.

        Returns:
            bool: True if operation is successful, False otherwise.
        """
        try:
            if "RANDOM_FOREST" in self.config:
                n_estimators = int(self.config["RANDOM_FOREST"]["n_estimators"])
                max_depth = self.config["RANDOM_FOREST"]["max_depth"]
                max_depth = int(max_depth) if max_depth != "None" else None
                min_samples_split = int(
                    self.config["RANDOM_FOREST"]["min_samples_split"]
                )
                min_samples_leaf = int(self.config["RANDOM_FOREST"]["min_samples_leaf"])
                self.log.info(
                    f"Using hyperparameters from config: n_estimators={n_estimators}, "
                    f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
                    f"min_samples_leaf={min_samples_leaf}"
                )
            else:
                n_estimators = 100
                max_depth = None
                min_samples_split = 2
                min_samples_leaf = 1
                self.log.info(
                    f"Using default hyperparameters: n_estimators={n_estimators}, "
                    f"max_depth={max_depth}, min_samples_split={min_samples_split}, "
                    f"min_samples_leaf={min_samples_leaf}"
                )

                self.config["RANDOM_FOREST"] = {
                    "n_estimators": str(n_estimators),
                    "max_depth": str(max_depth),
                    "min_samples_split": str(min_samples_split),
                    "min_samples_leaf": str(min_samples_leaf),
                    "path": self.model_path,
                }
                with open("config.ini", "w") as configfile:
                    self.config.write(configfile)

            categorical_cols = ["island", "sex"]
            numeric_cols = [
                "bill_length_mm",
                "bill_depth_mm",
                "flipper_length_mm",
                "body_mass_g",
            ]

            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, numeric_cols),
                    ("cat", categorical_transformer, categorical_cols),
                ],
                remainder="drop",
            )

            classifier = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            random_state=42,
                        ),
                    ),
                ]
            )

            y_train_flat = self.y_train.values.ravel()
            classifier.fit(self.X_train, y_train_flat)
            self.save_model(classifier, self.model_path)

            if predict:
                y_pred = classifier.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                report = classification_report(self.y_test, y_pred)
                self.log.info(f"Model accuracy: {accuracy:.4f}")
                self.log.info(f"Classification report:\n{report}")

            return True

        except Exception as e:
            self.log.error(f"Error in train_random_forest: {e}")
            self.log.error(traceback.format_exc())
            return False

    def save_model(self, classifier, path: str) -> bool:
        """
        Save a model to disk.

        Args:
            classifier: Trained model to save.
            path (str): Path to save the model to.

        Returns:
            bool: True if operation is successful, False otherwise.
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "wb") as f:
                pickle.dump(classifier, f)

            self.log.info(f"Model saved to {path}")
            return os.path.isfile(path)

        except Exception as e:
            self.log.error(f"Error in save_model: {e}")
            self.log.error(traceback.format_exc())
            return False


if __name__ == "__main__":
    classifier = PenguinClassifier()
    classifier.train_random_forest(predict=True)
