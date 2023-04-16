FROM python:3.8.16-slim-buster
WORKDIR /app

ADD src /app/src

# Copy application requirements file to the created working directory
COPY requirements.txt /app

# Install application dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements.txt


CMD ["python3", "/app/src/main.py"]

