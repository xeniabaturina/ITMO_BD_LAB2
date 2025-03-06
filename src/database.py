import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Create base class for declarative models
Base = declarative_base()

# Global engine and session factory, initialized lazily
engine = None
SessionLocal = None


class PredictionResult(Base):
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Input features
    culmen_length_mm = Column(Float)
    culmen_depth_mm = Column(Float)
    flipper_length_mm = Column(Float)
    body_mass_g = Column(Float)

    # Prediction
    predicted_species = Column(String)
    confidence = Column(Float)


def get_database_url():
    """Get database URL from environment variables"""
    # Get database connection details from environment variables
    DB_USER = os.getenv("POSTGRES_USER")
    DB_PASS = os.getenv("POSTGRES_PASSWORD")
    DB_NAME = os.getenv("POSTGRES_DB")
    DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
    DB_PORT = os.getenv("POSTGRES_PORT", "5432")

    # Create database URL
    return f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def init_db():
    """Initialize database connection and create tables"""
    global engine, SessionLocal

    # Check if we're in a test environment
    if os.environ.get("TESTING") == "1":
        # Use SQLite in-memory database for tests
        engine = create_engine("sqlite:///:memory:")
    else:
        # Use PostgreSQL for production
        engine = create_engine(get_database_url())

    # Create session factory
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)


def get_db():
    """Get database session"""
    # Initialize database if not already initialized
    if engine is None or SessionLocal is None:
        init_db()

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(
    db,
    culmen_length_mm: float,
    culmen_depth_mm: float,
    flipper_length_mm: float,
    body_mass_g: float,
    predicted_species: str,
    confidence: float,
):
    prediction = PredictionResult(
        culmen_length_mm=culmen_length_mm,
        culmen_depth_mm=culmen_depth_mm,
        flipper_length_mm=flipper_length_mm,
        body_mass_g=body_mass_g,
        predicted_species=predicted_species,
        confidence=confidence,
    )
    db.add(prediction)
    db.commit()
    return prediction
