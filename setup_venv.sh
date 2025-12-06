#!/bin/bash
# Setup mrb virtualenv with proper symlinks for Jetson CUDA packages
# Run this AFTER building TensorRT-LLM and installing torch from Jetson AI Lab

set -e

VENV_PATH="${HOME}/.virtualenvs/mrb"
USER_SITE="${HOME}/.local/lib/python3.10/site-packages"

echo "=========================================="
echo "mr-b Virtualenv Setup for Jetson"
echo "=========================================="

# Check if venv exists
if [ ! -d "${VENV_PATH}" ]; then
    echo "[1/5] Creating virtualenv at ${VENV_PATH}..."
    python3 -m venv "${VENV_PATH}"
else
    echo "[1/5] Virtualenv already exists at ${VENV_PATH}"
fi

# Activate
source "${VENV_PATH}/bin/activate"
VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")

echo "[2/5] Installing pure Python packages..."
pip install --quiet pynvml>=11.5.0
pip install --quiet 'numpy<2'

echo "[3/5] Removing any conflicting packages from venv..."
# Remove packages that might conflict with symlinked versions
rm -rf ${VENV_SITE}/torch* 2>/dev/null || true
rm -rf ${VENV_SITE}/cuda* 2>/dev/null || true
rm -rf ${VENV_SITE}/functorch 2>/dev/null || true
rm -rf ${VENV_SITE}/torchgen 2>/dev/null || true
rm -rf ${VENV_SITE}/torchvision* 2>/dev/null || true
rm -rf ${VENV_SITE}/torchaudio* 2>/dev/null || true
rm -rf ${VENV_SITE}/tensorrt* 2>/dev/null || true

echo "[4/5] Creating symlinks for CUDA packages..."

# PyTorch ecosystem (from Jetson AI Lab)
echo "  - torch 2.8.0"
ln -sf ${USER_SITE}/torch ${VENV_SITE}/
ln -sf ${USER_SITE}/torch-2.8.0.dist-info ${VENV_SITE}/
ln -sf ${USER_SITE}/torchgen ${VENV_SITE}/
ln -sf ${USER_SITE}/functorch ${VENV_SITE}/

echo "  - torchaudio 2.8.0"
ln -sf ${USER_SITE}/torchaudio ${VENV_SITE}/
ln -sf ${USER_SITE}/torchaudio-2.8.0.dist-info ${VENV_SITE}/ 2>/dev/null || true

echo "  - torchvision"
ln -sf ${USER_SITE}/torchvision ${VENV_SITE}/
ln -sf ${USER_SITE}/torchvision-*.dist-info ${VENV_SITE}/ 2>/dev/null || true
ln -sf ${USER_SITE}/torchvision.libs ${VENV_SITE}/ 2>/dev/null || true

echo "  - tensorrt_llm 0.12.0"
ln -sf ${USER_SITE}/tensorrt_llm ${VENV_SITE}/
ln -sf ${USER_SITE}/tensorrt_llm-*.dist-info ${VENV_SITE}/ 2>/dev/null || true

echo "  - cuda-python 12.6"
ln -sf ${USER_SITE}/cuda ${VENV_SITE}/
ln -sf ${USER_SITE}/cuda_python-12.6.dist-info ${VENV_SITE}/

echo "  - tensorrt (system)"
ln -sf /usr/lib/python3.10/dist-packages/tensorrt ${VENV_SITE}/
ln -sf /usr/lib/python3.10/dist-packages/tensorrt-*.dist-info ${VENV_SITE}/ 2>/dev/null || true
ln -sf /usr/lib/python3.10/dist-packages/tensorrt_dispatch ${VENV_SITE}/
ln -sf /usr/lib/python3.10/dist-packages/tensorrt_dispatch-*.dist-info ${VENV_SITE}/ 2>/dev/null || true
ln -sf /usr/lib/python3.10/dist-packages/tensorrt_lean ${VENV_SITE}/
ln -sf /usr/lib/python3.10/dist-packages/tensorrt_lean-*.dist-info ${VENV_SITE}/ 2>/dev/null || true

echo ""
echo "[5/5] Verifying imports..."
echo ""

python3 -c "import torch; print('✓ torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python3 -c "import torchvision; print('✓ torchvision:', torchvision.__version__)"
python3 -c "import torchaudio; print('✓ torchaudio:', torchaudio.__version__)" 2>/dev/null || echo "✗ torchaudio: not installed (optional)"
python3 -c "import tensorrt; print('✓ tensorrt:', tensorrt.__version__)"
python3 -c "import tensorrt_llm; print('✓ tensorrt_llm:', tensorrt_llm.__version__)"
python3 -c "from cuda import cudart; print('✓ cuda-python: OK')"
python3 -c "import numpy; print('✓ numpy:', numpy.__version__)"

echo ""
echo "=========================================="
echo "mrb virtualenv setup complete!"
echo ""
echo "Activate with: source ${VENV_PATH}/bin/activate"
echo "=========================================="
