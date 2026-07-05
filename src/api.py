"""FastAPI service that serves enrollment predictions from the trained model.

Run with::

    uvicorn src.api:app --reload

Then open http://127.0.0.1:8000/docs for interactive Swagger docs.

The service loads the pipeline saved by ``src.train`` once at startup. Because
preprocessing is baked into that pipeline, the endpoint accepts *raw* employee
attributes as JSON and applies the identical transformation used in training.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import config

app = FastAPI(
    title="Insurance Enrollment Prediction API",
    description="Predicts the probability an employee opts in to the voluntary "
    "insurance product, from demographic and employment attributes.",
    version="1.0.0",
)

# Loaded lazily so importing this module (e.g. in tests) does not require the
# model to exist yet. ``_get_model`` populates it on first use.
_model = None


def _get_model():
    """Load and cache the trained pipeline, with a clear error if it's missing."""
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
class Gender(str, Enum):
    female = "Female"
    male = "Male"
    other = "Other"


class MaritalStatus(str, Enum):
    single = "Single"
    married = "Married"
    divorced = "Divorced"
    widowed = "Widowed"


class EmploymentType(str, Enum):
    full_time = "Full-time"
    part_time = "Part-time"
    contract = "Contract"


class Region(str, Enum):
    west = "West"
    midwest = "Midwest"
    northeast = "Northeast"
    south = "South"


class HasDependents(str, Enum):
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


@app.post("/predict", response_model=PredictionResponse)
def predict(employee: EmployeeFeatures) -> PredictionResponse:
    """Predict enrollment for a single employee."""
    model = _get_model()
    # Build a one-row DataFrame with the exact column names the pipeline expects.
    row = pd.DataFrame([{
        "age": employee.age,
        "gender": employee.gender.value,
        "marital_status": employee.marital_status.value,
        "salary": employee.salary,
        "employment_type": employee.employment_type.value,
        "region": employee.region.value,
        "has_dependents": employee.has_dependents.value,
        "tenure_years": employee.tenure_years,
    }])
    proba = float(model.predict_proba(row)[0, 1])
    return PredictionResponse(enrolled=int(proba >= 0.5), probability=round(proba, 4))


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(employees: list[EmployeeFeatures]) -> list[PredictionResponse]:
    """Predict enrollment for a batch of employees in a single call."""
    model = _get_model()
    rows = pd.DataFrame([{
        "age": e.age,
        "gender": e.gender.value,
        "marital_status": e.marital_status.value,
        "salary": e.salary,
        "employment_type": e.employment_type.value,
        "region": e.region.value,
        "has_dependents": e.has_dependents.value,
        "tenure_years": e.tenure_years,
    } for e in employees])
    probas = model.predict_proba(rows)[:, 1]
    return [
        PredictionResponse(enrolled=int(p >= 0.5), probability=round(float(p), 4))
        for p in probas
    ]
