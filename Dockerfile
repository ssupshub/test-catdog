FROM python:3.7.8-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Add this line to start the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
