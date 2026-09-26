FROM python:3.11.3-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Same layout as the repo (/app/backend, /app/data) so relative paths behave identically.
WORKDIR /app/backend

COPY pyproject.toml .
COPY app ./app
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["sh", "-c", "python -m app.db.schema && uvicorn app.api.main:app --host 0.0.0.0 --port 8000"]
