FROM python:3.11-slim
LABEL authors="zofia"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

# Install project requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Obtain a copy of the python code and the diabetes model.
COPY app/ ./app/
COPY models/ ./models/

# Container expected to listen on port 8000.
# Verify successful connection through https://localhost:8000/
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]