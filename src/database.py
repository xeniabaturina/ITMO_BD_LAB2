import os
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Get database connection details from environment variables
DB_USER = os.getenv("POSTGRES_USER")
DB_PASS = os.getenv("POSTGRES_PASSWORD")
DB_NAME = os.getenv("POSTGRES_DB")
DB_HOST = os.getenv("POSTGRES_HOST", "postgres")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


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


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
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
