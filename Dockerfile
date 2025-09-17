# Dockerfile
FROM python:3.10-slim

# set workdir
WORKDIR /app

# system deps for pandas/SDV (may need adjustment for full SDV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc git libpq-dev curl wget \
    && rm -rf /var/lib/apt/lists/*

# copy files
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip
# install requirements
RUN pip wheel --no-deps --wheel-dir=/tmp/wheels -r /app/requirements.txt || true
RUN pip install --no-cache-dir -r /app/requirements.txt || true

COPY . /app

# Create a non-root user (recommended for Spaces)
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=7860
EXPOSE 7860

# start uvicorn on HOST 0.0.0.0 and $PORT
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
