FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY models ./models
COPY DATASETS ./DATASETS
COPY frontend/build ./frontend/build

EXPOSE 5001

CMD ["gunicorn", "--workers", "1", "--bind", "0.0.0.0:5001", "--timeout", "300", "backend.app:app"]
