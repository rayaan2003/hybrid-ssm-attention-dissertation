#!/bin/bash
# Environment setup for the JupyterHub "Data Science Projects" GPU container.
# Run this in a terminal inside JupyterHub (hub.a.comp-teach.qmul.ac.uk).

set -e

echo "Checking GPU..."
nvidia-smi

echo "Creating virtual environment..."
cd ~
virtualenv mamba_env
source mamba_env/bin/activate

echo "Installing PyTorch (CUDA 12.1 build)..."
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "Installing core dependencies..."
pip install transformers datasets accelerate wandb

echo "Installing Mamba's CUDA kernels (this requires a GPU)..."
pip install "causal-conv1d>=1.4.0"
pip install mamba-ssm

echo "Registering Jupyter kernel..."
pip install ipykernel
python -m ipykernel install --user --name=mamba_env --display-name "Python (mamba_env)"

echo ""
echo "Setup complete. Select 'Python (mamba_env)' as your kernel in Jupyter,"
echo "or run 'source ~/mamba_env/bin/activate' in a terminal before running scripts."
