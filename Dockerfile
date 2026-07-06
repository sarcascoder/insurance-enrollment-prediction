# Multi-stage build for the FastAPI prediction API.
#
#   Build:  docker build -t insurance-api .
#   Run:    docker run -p 8000:8000 insurance-api
#   Docs:   http://127.0.0.1:8000/docs
#
# Stage 1 trains the model in a full environment. Stage 2 is a slim runtime that
# copies ONLY the virtualenv, the source, and the trained model artifact — so the
# serving image does not carry the dataset, MLflow database, grid-search cruft, or
# build tooling. The container runs as a non-root user and exposes a healthcheck.

# --------------------------------------------------------------------------- #
# Stage 1 — builder: install deps and train the model
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN python -m venv /opt/venv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

# Produces models/model.joblib (the only build output the runtime needs).
RUN python -m src.train

# --------------------------------------------------------------------------- #
# Stage 2 — runtime: slim, non-root, model baked in
# --------------------------------------------------------------------------- #
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Non-root user for a smaller attack surface.
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/models/model.joblib ./models/model.joblib
COPY src/ ./src/

USER appuser
EXPOSE 8000

# Liveness probe hitting the app's own /health endpoint.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
