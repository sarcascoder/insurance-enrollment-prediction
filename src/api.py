"""FastAPI service that serves enrollment predictions from the trained model.

Run with::

    uvicorn src.api:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.

The service loads the pipeline saved by ``src.train`` once at startup. Because
preprocessing is baked into that pipeline, the endpoint accepts *raw* employee
attributes as JSON and applies the identical transformation used in training.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from enum import StrEnum

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config

logger = logging.getLogger(__name__)

# Module-level cache for the trained pipeline. Populated once at startup by the
# lifespan handler (so real requests never race on a lazy load), with a lazy
# fallback in ``_get_model`` for test clients that don't trigger startup.
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts (not per request)."""
    global _model
    if config.MODEL_PATH.exists():
        _model = joblib.load(config.MODEL_PATH)
        logger.info("Loaded model from %s", config.MODEL_PATH)
    else:
        logger.warning(
            "No model at %s; /predict will return 503 until `python -m src.train` runs.",
            config.MODEL_PATH,
        )
    yield
    _model = None  # release on shutdown


app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="Predicts the probability an employee opts in to the voluntary "
    "insurance product, from demographic and employment attributes.",
    version="1.0.0",
    lifespan=lifespan,
)


def _get_model():
    """Return the cached pipeline, loading lazily if startup didn't run (tests)."""
    global _model
    if _model is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Model artifact not found at '{config.MODEL_PATH}'. "
                    "Train it first with `python -m src.train`."
                ),
            )
        _model = joblib.load(config.MODEL_PATH)
    return _model


# --------------------------------------------------------------------------- #
# Request/response schemas
# --------------------------------------------------------------------------- #
# Enums mirror the categories seen in training. They give the API automatic
# validation and self-documenting Swagger dropdowns. Unknown categories would
# still be handled gracefully by the encoder, but rejecting them early is safer.
class Gender(StrEnum):
    female = "Female"
    male = "Male"
    other = "Other"


class MaritalStatus(StrEnum):
    single = "Single"
    married = "Married"
    divorced = "Divorced"
    widowed = "Widowed"


class EmploymentType(StrEnum):
    full_time = "Full-time"
    part_time = "Part-time"
    contract = "Contract"


class Region(StrEnum):
    west = "West"
    midwest = "Midwest"
    northeast = "Northeast"
    south = "South"


class HasDependents(StrEnum):
    yes = "Yes"
    no = "No"


class EmployeeFeatures(BaseModel):
    """One employee's attributes. Field constraints reject impossible inputs."""

    age: int = Field(..., ge=16, le=100, examples=[42])
    gender: Gender = Field(..., examples=[Gender.female])
    marital_status: MaritalStatus = Field(..., examples=[MaritalStatus.married])
    salary: float = Field(..., ge=0, examples=[85000.0])
    employment_type: EmploymentType = Field(..., examples=[EmploymentType.full_time])
    region: Region = Field(..., examples=[Region.west])
    has_dependents: HasDependents = Field(..., examples=[HasDependents.yes])
    tenure_years: float = Field(..., ge=0, le=60, examples=[6.5])


class PredictionResponse(BaseModel):
    enrolled: int = Field(..., description="Predicted label: 1=enroll, 0=not")
    probability: float = Field(..., description="Probability of enrollment (0-1)")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health() -> dict:
    """Liveness/readiness check: reports whether the model is loadable."""
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


def _predict(employees: list[EmployeeFeatures]) -> list[PredictionResponse]:
    """Shared inference path for both single and batch endpoints.

    ``model_dump(mode="json")`` turns each validated record into a dict whose
    keys already match the training feature names and whose enum values are
    plain strings — so the DataFrame columns line up with the pipeline without
    any manual field-by-field mapping (which would silently rot if a column is
    added).
    """
    model = _get_model()
    frame = pd.DataFrame([e.model_dump(mode="json") for e in employees])
    probas = model.predict_proba(frame)[:, 1]
    return [
        PredictionResponse(enrolled=int(p >= 0.5), probability=round(float(p), 4)) for p in probas
    ]


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeFeatures) -> PredictionResponse:
    """Predict enrollment for a single employee."""
    return _predict([employee])[0]


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(employees: list[EmployeeFeatures]) -> list[PredictionResponse]:
    """Predict enrollment for a batch of employees in a single call."""
    return _predict(employees)
