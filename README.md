# mini-llm

A minimal implementation for fine-tuning small language models using Hugging Face's Transformers library.

## Project Overview

This project provides a simple framework for fine-tuning GPT-2 on the Guanaco dataset, demonstrating how to:
- Load datasets from Hugging Face's Hub
- Configure a pre-trained model for fine-tuning
- Train the model using Hugging Face's Trainer API

## Hardware Requirements

This project has been tested with:
- NVIDIA RTX 4060 8GB
- WSL2 (Windows Subsystem for Linux)
- Docker with NVIDIA Container Toolkit

### WSL2 + Docker GPU Setup

1. Ensure WSL2 is installed and configured with GPU access:
   ```bash
   wsl --update
   ```

2. Install NVIDIA Container Toolkit for Docker:
   ```bash
   # Inside WSL2
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

3. Verify GPU is accessible:
   ```bash
   docker run --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```

### Memory Optimization

For RTX 4060 8GB, consider these optimizations:
- Reduced batch size (already set to 1)
- Using gradient checkpointing if needed
- Gradient accumulation for larger effective batch sizes

## Installation

### Local Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/mini-llm.git
cd mini-llm

# Install dependencies
pip install -r requirements.txt
```

### Docker Setup

```bash
# Build the Docker image
docker build -t mini-llm .

# Run the container (basic)
docker run --gpus all mini-llm

# Run with memory optimization for RTX 4060 8GB
docker run --gpus all --shm-size=1g --ulimit memlock=-1 mini-llm
```

## Usage

Run the fine-tuning script:

```bash
# Basic usage
python scripts/finetune.py

# For RTX 4060 8GB optimized memory usage
python scripts/finetune.py --gradient_checkpointing --gradient_accumulation_steps 4
```

## Model and Dataset

- **Base Model**: GPT-2
- **Dataset**: [mlabonne/guanaco-llama2-1k](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)
- **Training Parameters**:
  - Batch size: 1
  - Epochs: 1
  - Precision: FP16

## Project Structure

```
mini-llm/
├── Dockerfile        # CUDA-enabled environment setup
├── README.md         # This file
├── requirements.txt  # Python dependencies
└── scripts/
    └── finetune.py   # Main fine-tuning script
```

## Checkpoints

The fine-tuned model checkpoints will be saved in the `./checkpoints` directory.
