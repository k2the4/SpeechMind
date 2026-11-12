FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Main environment: Whisper, Streamlit, WavLM and MuRIL.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements-space.txt /app/requirements-space.txt
RUN pip install --no-cache-dir -r /app/requirements-space.txt

# Isolated environment: Qwen3-ASR and IndicTrans2 require a newer
# Transformers version than the main SpeechMind runtime.
RUN python -m venv /app/.venv_srota
COPY requirements_hinglish_asr.txt /app/requirements_hinglish_asr.txt
COPY requirements_indic_translation.txt /app/requirements_indic_translation.txt
RUN /app/.venv_srota/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && /app/.venv_srota/bin/pip install --no-cache-dir -r /app/requirements_hinglish_asr.txt \
    && /app/.venv_srota/bin/pip install --no-cache-dir -r /app/requirements_indic_translation.txt

COPY . /app

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]
