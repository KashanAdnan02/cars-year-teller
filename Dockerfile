FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for some ML libraries)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Install production server
RUN pip install gunicorn

# Copy the rest of your code
COPY . .

# Expose port
EXPOSE 8080

# Run with Gunicorn (recommended for production)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]