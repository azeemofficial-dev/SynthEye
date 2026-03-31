FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash syntheye \
    && mkdir -p /app/data /app/logs \
    && chown -R syntheye:syntheye /app

USER syntheye

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SYNTHEYE_ENV=production
ENV SYNTHEYE_REQUIRE_AUTH=1
ENV SYNTHEYE_SESSION_TTL_SECONDS=86400
ENV SYNTHEYE_SESSION_HTTPS_ONLY=1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]
