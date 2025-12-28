# Use Python slim image
FROM python:3.11-slim

# Install system dependencies for pdf2image and pylibdmtx
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libdmtx0b \
    libdmtx-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for Docker cache)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .

# Expose port (Render will set PORT env var)
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
