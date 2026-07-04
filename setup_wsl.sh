#!/bin/bash
set -e

# 1. Copy project to WSL native filesystem
cd ~
cp -r /mnt/e/project/Growable-LLM . 2>/dev/null || true
cd Growable-LLM

# 2. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Add uv to PATH
export PATH="$HOME/.local/bin:$PATH"

# 4. Install dependencies with Python 3.12
uv python install 3.12 2>/dev/null || true
uv sync --python 3.12 --extra benchmark --extra experiment

# 5. Verify PyTorch + CUDA
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB')"