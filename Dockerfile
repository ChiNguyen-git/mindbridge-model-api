FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for PyTorch
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Expose the port your app runs on
EXPOSE 5000

# Command to run your application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app", "--timeout", "120"]