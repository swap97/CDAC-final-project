FROM python:3.9-slim

# Installing system dependencies for OpenCV and MTCNN
RUN apt-get update --fix-missing && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Installing python dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY best_deepfake_model.h5 .
COPY app.py .

# Port Flask runs on
EXPOSE 5000

# Setting up the environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Command to run the application using Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
