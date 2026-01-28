# Base image with PyTorch and CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory inside the container
WORKDIR /app

# Install system dependencies (git is often needed for installing unsloth)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the API runs on
EXPOSE 8000

# Command to run the application
# We use 'python -m' to ensure sys.path is correct
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
