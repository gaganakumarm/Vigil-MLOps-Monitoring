# Use the Python version defined in the .env file
ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION}-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from writing .pyc files and buffer I/O
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir requests

# Copy the rest of the application code
COPY . .

# Expose the default FastAPI port
EXPOSE 8000

# Command to run the API service with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
