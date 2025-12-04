#!/bin/bash
# Build CTranslate2 with CUDA support for Jetson JetPack 6
# This is required because the pip wheel doesn't include CUDA support for aarch64
set -e

CTRANSLATE_VERSION="${CTRANSLATE_VERSION:-4.5.0}"
CTRANSLATE_SOURCE="/tmp/CTranslate2"

echo "=== Building CTranslate2 ${CTRANSLATE_VERSION} with CUDA for Jetson ==="

# Setup CUDA paths for JetPack
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc not found at /usr/local/cuda/bin/"
    echo "Make sure CUDA is installed (comes with JetPack)."
    echo "Try: ls /usr/local/cuda/bin/nvcc"
    exit 1
fi

echo "Found CUDA: $(nvcc --version | head -n4 | tail -n1)"

# Install build dependencies
echo "Installing build dependencies..."
sudo apt-get update
sudo apt-get install -y \
    cmake \
    build-essential \
    git

# cuDNN is already installed on JetPack - verify it exists
if [ ! -f /usr/include/cudnn.h ] && [ ! -f /usr/local/cuda/include/cudnn.h ]; then
    echo "WARNING: cudnn.h not found, trying to install cudnn-dev..."
    sudo apt-get install -y libcudnn9-dev 2>/dev/null || \
    sudo apt-get install -y libcudnn8-dev 2>/dev/null || \
    echo "cuDNN dev headers not found in apt, assuming JetPack has them"
fi

# Uninstall existing ctranslate2 (pip version without CUDA)
echo "Removing existing ctranslate2..."
pip uninstall -y ctranslate2 2>/dev/null || true

# Clone CTranslate2
echo "Cloning CTranslate2 v${CTRANSLATE_VERSION}..."
rm -rf ${CTRANSLATE_SOURCE}
git clone --branch=v${CTRANSLATE_VERSION} --recursive https://github.com/OpenNMT/CTranslate2.git ${CTRANSLATE_SOURCE}

# Build C++ library
echo "Building CTranslate2 C++ library..."
mkdir -p ${CTRANSLATE_SOURCE}/build
cd ${CTRANSLATE_SOURCE}/build

cmake .. \
    -DWITH_CUDA=ON \
    -DWITH_CUDNN=ON \
    -DWITH_MKL=OFF \
    -DOPENMP_RUNTIME=COMP \
    -DCMAKE_INSTALL_PREFIX=/usr/local

make -j$(nproc)
sudo make install
sudo ldconfig

# Save current PyTorch version before it gets overwritten
TORCH_VERSION=$(pip show torch 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
echo "Current PyTorch version: ${TORCH_VERSION:-not installed}"

# Build Python wheel
echo "Building Python wheel..."
cd ${CTRANSLATE_SOURCE}/python

# Install requirements but skip torch (we have Jetson's CUDA-enabled version)
pip install --no-cache-dir pybind11 setuptools wheel numpy

# Build and install wheel
python setup.py bdist_wheel
pip install --no-cache-dir --no-deps dist/ctranslate2*.whl

# Reinstall Jetson PyTorch if it was overwritten
TORCH_VERSION_NOW=$(pip show torch 2>/dev/null | grep "^Version:" | cut -d' ' -f2)
if [ "$TORCH_VERSION" != "$TORCH_VERSION_NOW" ] || ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "PyTorch CUDA support lost, reinstalling Jetson PyTorch..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    # JetPack 6.x PyTorch from Jetson AI Lab (has cuDNN 9 support)
    pip install --no-cache-dir torch --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
fi

# Verify installation
echo ""
echo "=== Verifying CTranslate2 CUDA support ==="
python3 -c "
import ctranslate2
print(f'CTranslate2 version: {ctranslate2.__version__}')
cuda_devices = ctranslate2.get_cuda_device_count()
print(f'CUDA devices found: {cuda_devices}')
if cuda_devices > 0:
    print('CUDA support: OK')
else:
    print('WARNING: No CUDA devices found')
"

echo ""
echo "=== CTranslate2 with CUDA built successfully! ==="
