FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

WORKDIR /app

# Install torch first (CUDA-matched)
RUN python3 -m pip install --no-cache-dir torch==2.4.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# Install build deps that flash-attn's setup needs
RUN python3 -m pip install --no-cache-dir packaging ninja psutil setuptools wheel

# Now compile flash-attn (limit jobs to avoid RAM explosion)
RUN MAX_JOBS=4 python3 -m pip install --no-cache-dir flash-attn==2.6.3 --no-build-isolation

# Now copy the rest of the app
COPY requirements.txt .
COPY app.py .
#COPY reference_voice.wav .

# Explicitly install pydub + the rest from requirements.txt
RUN python3 -m pip install --no-cache-dir pydub==0.25.1
RUN python3 -m pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]