# Use the official Python base image
FROM python:3.11.0-slim

# Set the working directory inside the container
WORKDIR /app

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Copy the requirements file to the working directory
COPY server/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY models .
COPY losses.py .

WORKDIR /app/server

COPY server/* .

# Expose the port on which the application will run
EXPOSE 8080

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8080"]