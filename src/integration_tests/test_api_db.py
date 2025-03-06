import json
import pytest
from src.api import app
from src.database import Base, init_db
from sqlalchemy import create_engine

# Test database configuration
TEST_DB_USER = "test_user"
TEST_DB_PASS = "test_pass"
TEST_DB_NAME = "test_db"
TEST_DB_HOST = "localhost"
TEST_DB_PORT = "5433"

# Create test database URL
TEST_DATABASE_URL = f"postgresql://{TEST_DB_USER}:{TEST_DB_PASS}@{TEST_DB_HOST}:{TEST_DB_PORT}/{TEST_DB_NAME}"


@pytest.fixture
def client():
    """Create a test client"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture(scope="function")
def test_db():
    """Set up test database"""
    # Set test environment variables for the Flask app
    import os

    os.environ["POSTGRES_USER"] = TEST_DB_USER
    os.environ["POSTGRES_PASSWORD"] = TEST_DB_PASS
    os.environ["POSTGRES_DB"] = TEST_DB_NAME
    os.environ["POSTGRES_HOST"] = TEST_DB_HOST
    os.environ["POSTGRES_PORT"] = TEST_DB_PORT

    # Create test engine with explicit connection string
    engine = create_engine(TEST_DATABASE_URL)

    # Create all tables
    Base.metadata.create_all(engine)

    # Initialize the database
    init_db()

    yield

    # Cleanup
    Base.metadata.drop_all(engine)


def test_predict_endpoint(client, test_db):
    """Test the predict endpoint with database integration"""
    # Test data
    test_data = {
        "island": "Torgersen",
        "bill_length_mm": 39.1,
        "bill_depth_mm": 18.7,
        "flipper_length_mm": 181.0,
        "body_mass_g": 3750.0,
        "sex": "MALE",
    }

    # Make prediction request
    response = client.post(
        "/predict", data=json.dumps(test_data), content_type="application/json"
    )

    # Check response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "predicted_species" in data
    assert "probabilities" in data


def test_predictions_endpoint(client, test_db):
    """Test the predictions endpoint"""
    # First make some predictions
    test_data = [
        {
            "island": "Torgersen",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181.0,
            "body_mass_g": 3750.0,
            "sex": "MALE",
        },
        {
            "island": "Dream",
            "bill_length_mm": 42.3,
            "bill_depth_mm": 19.2,
            "flipper_length_mm": 190.0,
            "body_mass_g": 4100.0,
            "sex": "FEMALE",
        },
    ]

    # Make predictions
    for data in test_data:
        response = client.post(
            "/predict", data=json.dumps(data), content_type="application/json"
        )
        assert response.status_code == 200

    # Test getting predictions
    response = client.get("/predictions")

    # Check response
    assert response.status_code == 200
    data = json.loads(response.data)
    assert data["success"] is True
    assert "predictions" in data
    assert len(data["predictions"]) == 2  # We made 2 predictions

    # Test pagination
    response = client.get("/predictions?limit=1&offset=1")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert len(data["predictions"]) == 1
