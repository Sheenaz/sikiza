# Hugging Face Spaces variant of serving/Dockerfile: same image, but Spaces requires
# port 7860 and a non-root user with uid 1000. State (escalation DB, audit log) is ephemeral here.
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
COPY artifacts/model/ artifacts/model/

RUN useradd --create-home --uid 1000 triage && mkdir -p /app/state/logs && chown -R triage /app/state
USER triage
# The coping library is still a counsellor-unapproved draft; COUNSELLOR_API_KEY is set as a Space secret.
ENV TRIAGE_AUDIT_LOG=/app/state/logs/predictions.jsonl TRIAGE_ESCALATION_DB=/app/state/escalations.sqlite \
    TRIAGE_MODEL_DIR=/app/artifacts/model ALLOW_DRAFT_LIBRARY=1

EXPOSE 7860
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
