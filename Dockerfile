FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY api/ ./api/

# Expose port
EXPOSE 8000

# Run
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
