FROM python:3.11.3-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# Same layout as the repo (/app/backend, /app/data) so relative paths behave identically.
WORKDIR /app/backend

# Dependencies first, against an empty package, so this layer is cached until pyproject.toml changes.
# (Copying the code first made every code change re-download everything from PyPI, which timed out.)
COPY pyproject.toml .
RUN mkdir app && pip install --no-cache-dir -e .
COPY app ./app

EXPOSE 8000
CMD ["sh", "-c", "python -m app.db.schema && uvicorn app.api.main:app --host 0.0.0.0 --port 8000"]
