# Nihai Algoritma — sinyal-only canlı sistem + web pano (D2).
# 7/24 çalıştırıldığında PREDICT_TRACK=true ile canlı forecast'i kapalı-döngü kaydeder
# (docs/EDGE_FINDINGS.md: survivorship-temiz OOS kenar sorusunun tek ücretsiz kanıt yolu).
FROM python:3.12-slim

# Agg → headless matplotlib (B2 /grafik); NY borsa saati; tamponsuz log
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    TZ=America/New_York

WORKDIR /app

# numpy/scipy/pandas/matplotlib/lxml için derleme araçları + tzdata (borsa takvimi) + curl (healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc tzdata curl \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları önce kur → kod değişince katman önbelleği korunur
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Kalıcı çıktı dizinleri (compose volume ile host'a bağlanır)
RUN mkdir -p /app/data /app/logs

EXPOSE 8765

# Konteyner içinden erişilebilirlik için web sunucusu 0.0.0.0'a bağlanmalı (compose'ta WEB_HOST=0.0.0.0)
HEALTHCHECK --interval=60s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -fsS http://127.0.0.1:8765/api/status || exit 1

CMD ["python", "-m", "nihai.execution.main", "--serve-web"]
