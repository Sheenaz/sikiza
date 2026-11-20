# Torch-free serving image: onnxruntime + tokenizers only (~350 MB).
# Build from the project root:
#   docker build -f serving/Dockerfile -t sheng-triage:0.1 .
# Host on Kenya-based infrastructure (DPA 2019 data localisation for health data).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY triage/ triage/
COPY serving/ serving/
COPY configs/ configs/
COPY data/lexicon/ data/lexicon/
COPY data/library/ data/library/
# The pinned ONNX int8 bundle produced by scripts/export_onnx.py
COPY artifacts/model/ artifacts/model/

# Model files stay root-owned and read-only (no recursive chown: that would duplicate the
# 280 MB model layer). The app user can write only logs and the escalation DB under /app/state.
RUN useradd --create-home --uid 10001 triage && mkdir -p /app/state/logs && chown -R triage /app/state
USER triage
ENV TRIAGE_AUDIT_LOG=/app/state/logs/predictions.jsonl TRIAGE_ESCALATION_DB=/app/state/escalations.sqlite
VOLUME ["/app/state"]

ENV TRIAGE_MODEL_DIR=/app/artifacts/model
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
