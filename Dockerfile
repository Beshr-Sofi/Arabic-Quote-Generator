FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd -m appuser \
    && mkdir -p /home/appuser/.cache/huggingface \
    && chown -R appuser:appuser /home/appuser

COPY --chown=appuser . .

USER appuser

ENV HF_HOME=/home/appuser/.cache/huggingface \
    HF_HUB_CACHE=/home/appuser/.cache/huggingface/hub \
    MODEL_PATH=models/final

RUN python -m src.modeling.pull_model

EXPOSE 7860