FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install base tools
RUN apt update && apt install -y git python3-pip

# Set up Python environment
RUN pip3 install --upgrade pip

# Copy your project files
WORKDIR /app
COPY . .

# Install Python dependencies
# RUN pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
RUN pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers datasets accelerate flask

CMD ["python3", "scripts/finetune.py"]
