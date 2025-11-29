# Demo Dockerfile for Vehicle Predictive Maintenance app (Streamlit)
FROM python:3.10-slim

# Avoid prompts from apt
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Default command: run streamlit demo app
CMD ["streamlit", "run", "app/dashboard_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]