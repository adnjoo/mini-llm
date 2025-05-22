FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install base tools
RUN apt update && apt install -y git python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy project files
WORKDIR /app
COPY . .

# Install Python dependencies in one layer, then clean up cache
RUN pip install torch==2.2.2+cu121 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 && \
    pip install "numpy<2.0.0" transformers datasets accelerate flask && \
    rm -rf /root/.cache/pip

# Optional: Prevent Hugging Face cache bloat (removes downloaded model/tokenizer cache)
RUN rm -rf /root/.cache/huggingface

# Run training script
CMD ["python3", "scripts/finetune.py"]
