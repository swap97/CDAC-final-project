FROM python:3.9-slim

# Installing system dependencies for OpenCV and MTCNN
RUN apt-get update --fix-missing && \
    for i in 1 2 3; do \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && break || sleep 5; \
    done && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Installing python dependencies
COPY --chown=user best_deepfake_model.h5 .
COPY --chown=user app.py .

# Port Flask runs on
EXPOSE 7860

# Setting up the environment variables
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Command to run the application using Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
