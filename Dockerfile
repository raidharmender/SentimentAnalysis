FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy project files for dependency installation
COPY pyproject.toml uv.lock ./

# Install uv and dependencies
RUN pip install uv && uv sync --frozen

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p uploads processed

# Expose ports
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "main.py", "--mode", "both", "--host", "0.0.0.0"] 