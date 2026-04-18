FROM python:3.14-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
ENTRYPOINT [ "python", "terrain_estimator.py" ]
