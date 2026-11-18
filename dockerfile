# Use modern Python 3.11 base image (required for numpy 2.0, scipy 1.14, and scikit-learn 1.6 precompiled wheels)
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (libgomp1 is required for OpenMP in XGBoost/Scikit-Learn, curl for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*

# Upgrade pip and wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy the requirements file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the backend code into the container
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Command to run the application
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
