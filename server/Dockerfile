# Use official Python slim image
FROM python:3.10-slim

# Install curl
RUN apt-get update && apt-get install -y curl

# Set workdir
WORKDIR /app

# Create the static folder explicitly
RUN mkdir -p /app/static

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Set environment variables so Flask knows where the app is and which port to use
ENV FLASK_APP=server.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Optionally add a health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8080/api/test || exit 1

# Expose port
EXPOSE 8080

# Run the server
CMD ["flask", "run"]
