FROM python:3.13-slim

WORKDIR /app
ARG INSTALL_CRAWLER_DEPS=false

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV CRAWL4_AI_BASE_DIRECTORY=/app/.crawl4ai-data

COPY requirements.txt requirements-ml.txt requirements-crawlers.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && if [ "$INSTALL_CRAWLER_DEPS" = "true" ]; then apt-get install -y --no-install-recommends \
    curl \
    libnss3 \
    libatk-bridge2.0-0 \
    libdrm2 \
    libxkbcommon0 \
    libgtk-3-0 \
    libgbm1 \
    libasound2 \
    libxshmfence1 \
    libx11-xcb1 \
    libxcb1 \
    libxcomposite1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxrender1 \
    libxi6 \
    libxtst6 \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libdbus-1-3 \
    tesseract-ocr \
    tesseract-ocr-kor \
    tesseract-ocr-eng; \
    fi \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-ml.txt
RUN if [ "$INSTALL_CRAWLER_DEPS" = "true" ]; then \
    pip install --no-cache-dir -r requirements-crawlers.txt && \
    python -m playwright install chromium; \
    fi

COPY . .

EXPOSE 8000
