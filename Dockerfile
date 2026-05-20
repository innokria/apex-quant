FROM python:3.10-slim

# -----------------------------
# SYSTEM
# -----------------------------
RUN apt-get update && apt-get install -y \
    git wget curl bash build-essential cmake dos2unix \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -----------------------------
# PYTHON
# -----------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# ENV
# -----------------------------
ENV HF_HOME=/tmp/hf_cache
ENV APEX_DIR=/app/apex-quant
ENV PATH="/usr/local/bin:$PATH"
ENV LLAMA_QUANTIZE=/usr/local/bin/llama-quantize

RUN mkdir -p /tmp/hf_cache

# -----------------------------
# llama-quantize (ONLY RELIABLE PART)
# -----------------------------
RUN curl -L -o /usr/local/bin/llama-quantize \
    https://github.com/ggerganov/llama.cpp/releases/download/b4960/llama-quantize && \
    chmod +x /usr/local/bin/llama-quantize

# -----------------------------
# apex-quant (ONLY FOR PROFILES)
# -----------------------------
RUN git clone --depth 1 https://github.com/innokria/apex-quant.git /app/apex-quant

RUN chmod +x /app/apex-quant/scripts/quantize.sh || true

# -----------------------------
# APP
# -----------------------------
COPY app.py /app/app.py

EXPOSE 7860

CMD ["python3", "app.py"]
