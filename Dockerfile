# Container image that serves the FastAPI prediction API.
#
# Build:  docker build -t insurance-api .
# Run:    docker run -p 8000:8000 insurance-api
# Then open http://127.0.0.1:8000/docs
#
# The image bakes a freshly trained model into the layer at build time, so the
# container is self-contained and needs no volume mounts to serve predictions.

FROM python:3.12-slim

# Avoid interactive prompts and keep Python output unbuffered for clean logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source, dataset, and configuration.
COPY src/ ./src/
COPY data/ ./data/

# Train the model at build time so the artifact ships inside the image.
RUN python -m src.train

EXPOSE 8000

# 0.0.0.0 so the server is reachable from outside the container.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
