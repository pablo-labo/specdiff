#!/bin/bash
set -euo pipefail

# Usage:
#   bash start.sh
#   TORCH_CUDA=cu118 bash start.sh
#   CONDA_ENV=spec bash start.sh

VENV_DIR=".venv"
CONDA_ENV="${CONDA_ENV:-spec}"
TORCH_CUDA="${TORCH_CUDA:-cu121}" # RTX 3090 default recommendation
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"

activate_python_env() {
  if command -v conda >/dev/null 2>&1; then
    echo "Conda detected. Preparing environment: ${CONDA_ENV}"
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"

    if ! conda run -n "${CONDA_ENV}" python -V >/dev/null 2>&1; then
      echo "Creating conda env ${CONDA_ENV} (python=3.10)..."
      conda create -n "${CONDA_ENV}" python=3.10 -y
    fi

    conda activate "${CONDA_ENV}"
    echo "Activated conda env: ${CONDA_ENV}"
  else
    echo "Conda not found. Falling back to venv: ${VENV_DIR}"
    if [ ! -d "${VENV_DIR}" ]; then
      echo "Creating virtual environment at ${VENV_DIR}..."
      python3 -m venv "${VENV_DIR}"
    fi
    # shellcheck disable=SC1091
    source "${VENV_DIR}/bin/activate"
    echo "Activated venv: ${VENV_DIR}"
  fi
}

activate_python_env

echo "Upgrading pip/setuptools/wheel..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing PyTorch with CUDA (${TORCH_CUDA})..."
pip install --index-url "${TORCH_INDEX_URL}" torch torchvision torchaudio

echo "Installing project dependencies..."
pip install -r requirements.txt

echo "Checking CUDA availability..."
python -c "import torch; print('torch=', torch.__version__); print('cuda_available=', torch.cuda.is_available()); print('cuda=', torch.version.cuda); print('gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 登录 HuggingFace (如果模型是私有的，Qwen2.5 通常不需要这一步)
# huggingface-cli login --token YOUR_TOKEN

echo "Starting Simulation..."
python main.py

# (可选) 如果你想运行 Web UI
# echo "Starting Web UI..."
# python app.py
