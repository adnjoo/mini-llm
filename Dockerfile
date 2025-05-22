FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install base tools
RUN apt update && apt install -y git python3-pip

# Upgrade pip
RUN pip3 install --upgrade pip

# Copy project files
WORKDIR /app
COPY . .

# Install PyTorch with CUDA 12.1
RUN pip install torch==2.2.2+cu121 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# FIX: Downgrade NumPy for compatibility
RUN pip install "numpy<2.0.0"

# Install Hugging Face + Flask
RUN pip install transformers datasets accelerate flask

# Run training script
CMD ["python3", "scripts/finetune.py"]
