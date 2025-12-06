#!/bin/bash
# TensorRT-LLM Installation for Jetson Orin
# JetPack 6.x / L4T R36.x
#
# Based on: https://github.com/NVIDIA/TensorRT-LLM/blob/v0.12.0-jetson/README4Jetson.md

set -e

echo "========================================"
echo "TensorRT-LLM Setup for Jetson Orin"
echo "========================================"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRTLLM_DIR="${SCRIPT_DIR}/TensorRT-LLM"

# Orin architecture
CUDA_ARCH=87

echo "[1/8] Installing system prerequisites..."
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev git-lfs ccache

echo ""
echo "[2/8] Installing cuSPARSELt..."
if [ ! -f "/usr/local/cuda/lib64/libcusparseLt.so" ]; then
    wget -q https://raw.githubusercontent.com/pytorch/pytorch/9b424aac1d70f360479dd919d6b7933b5a9181ac/.ci/docker/common/install_cusparselt.sh
    export CUDA_VERSION=12.6
    sudo -E bash ./install_cusparselt.sh
    rm -f install_cusparselt.sh
else
    echo "cuSPARSELt already installed"
fi

echo ""
echo "[3/8] Installing Python dependencies..."
# NumPy <2 required - torch 2.8.0 was compiled with NumPy 1.x
pip3 install "numpy<2" --force-reinstall

# cuda-python 12.x required for cudart import compatibility
pip3 install cuda-python==12.6

echo ""
echo "[4/8] Installing PyTorch from Jetson AI Lab..."
# MUST use --index-url (not --extra-index-url) to avoid pulling CPU-only torch from PyPI
pip3 install torch==2.8.0 torchaudio --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir

# Verify CUDA torch - CRITICAL: bindings must be compiled against CUDA-enabled torch
echo "Verifying torch CUDA support..."
TORCH_CUDA_CHECK=$(python3 -c "import torch; print('OK' if torch.cuda.is_available() else 'FAIL')")
if [ "${TORCH_CUDA_CHECK}" != "OK" ]; then
    echo "ERROR: torch.cuda.is_available() returned False!"
    echo "TensorRT-LLM bindings MUST be compiled against CUDA-enabled torch."
    echo ""
    echo "Possible fixes:"
    echo "  1. Ensure you're not in a virtualenv: deactivate"
    echo "  2. Remove CPU-only torch: pip3 uninstall torch"
    echo "  3. Reinstall: pip3 install torch==2.8.0 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir"
    exit 1
fi
python3 -c "import torch; print(f'✓ torch {torch.__version__} with CUDA {torch.version.cuda}')"

echo ""
echo "[5/8] Cloning TensorRT-LLM (jetson branch)..."
if [ ! -d "$TRTLLM_DIR" ]; then
    git clone https://github.com/NVIDIA/TensorRT-LLM.git "$TRTLLM_DIR"
    cd "$TRTLLM_DIR"
    git checkout v0.12.0-jetson
else
    echo "TensorRT-LLM already cloned"
    cd "$TRTLLM_DIR"
    git checkout v0.12.0-jetson
fi

# Patch requirements to use torch 2.8.0 and compatible typing-extensions
echo "Patching requirements for torch 2.8.0 compatibility..."
sed -i 's|torch @ https://developer.download.nvidia.cn/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl|torch==2.8.0|' requirements-jetson.txt
sed -i 's|typing-extensions==4.8.0|typing-extensions>=4.10.0|g' requirements-dev-jetson.txt requirements-dev.txt requirements-dev-windows.txt

# Patch pybind CMakeLists.txt (pybind11 NewTools requires find_package(Python) not Python3)
echo "Patching pybind CMakeLists.txt for Python Development.Module..."
sed -i '/^find_package(pybind11 REQUIRED)/i # Fix for pybind11 NewTools requiring Python (not Python3) with Development.Module\nfind_package(Python COMPONENTS Interpreter Development.Module REQUIRED)' cpp/tensorrt_llm/pybind/CMakeLists.txt

# Patch parse_make_options call (fails when TORCH_CXX_FLAGS is empty)
echo "Patching CMakeLists.txt for empty TORCH_CXX_FLAGS..."
sed -i 's|parse_make_options(${TORCH_CXX_FLAGS} "TORCH_CXX_FLAGS")|if(TORCH_CXX_FLAGS)\n      parse_make_options(${TORCH_CXX_FLAGS} "TORCH_CXX_FLAGS")\n    endif()|' cpp/CMakeLists.txt

# Downgrade pybind11 (3.x has breaking CMake changes with python_add_library)
echo "Ensuring pybind11 2.x is installed (3.x has CMake incompatibilities)..."
pip3 install 'pybind11>=2.6,<3.0' --force-reinstall

echo ""
echo "[6/8] Initializing submodules (CUTLASS, etc.)..."
git submodule update --init --recursive

# Checkout the specific CUTLASS commit expected by this branch
echo "Checking out correct CUTLASS version..."
cd 3rdparty/cutlass
CUTLASS_COMMIT=$(cd "${TRTLLM_DIR}" && git ls-tree HEAD 3rdparty/cutlass | awk '{print $3}')
git checkout ${CUTLASS_COMMIT}

# Create setup.py wrapper (newer CUTLASS doesn't have one)
if [ ! -f python/setup.py ]; then
    echo "Creating CUTLASS python/setup.py wrapper..."
    cat > python/setup.py << 'SETUPEOF'
from setup_library import perform_setup
if __name__ == "__main__":
    perform_setup()
SETUPEOF
fi
cd "${TRTLLM_DIR}"

echo ""
echo "[7/8] Pulling LFS files..."
git lfs pull

echo ""
echo "[8/8] Building TensorRT-LLM wheel (this will take a while)..."
echo "      CUDA architecture: ${CUDA_ARCH} (Orin)"
echo ""

# Final torch verification before build - this is critical!
echo "Pre-build torch verification..."
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
TORCH_CUDA=$(python3 -c "import torch; print(torch.cuda.is_available())")
TORCH_PATH=$(python3 -c "import torch; print(torch.__file__)")
echo "  torch version: ${TORCH_VERSION}"
echo "  CUDA available: ${TORCH_CUDA}"
echo "  torch path: ${TORCH_PATH}"

if [ "${TORCH_CUDA}" != "True" ]; then
    echo ""
    echo "FATAL: torch.cuda.is_available() is False!"
    echo "The TensorRT-LLM bindings will be compiled against this torch."
    echo "If CUDA is not available, the bindings will fail at runtime with:"
    echo "  'undefined symbol: _Z16THPVariable_WrapN2at10TensorBaseE'"
    echo ""
    echo "Please ensure CUDA-enabled torch 2.8.0 from Jetson AI Lab is installed."
    exit 1
fi

python3 scripts/build_wheel.py \
    --clean \
    --cuda_architectures ${CUDA_ARCH} \
    -DENABLE_MULTI_DEVICE=0 \
    --build_type Release \
    --benchmarks \
    --use_ccache

echo ""
echo "Installing wheel..."
pip3 install build/tensorrt_llm-*.whl --force-reinstall

echo ""
echo "Verifying installation..."
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')" || {
    echo "Import failed!"
    exit 1
}

echo ""
echo "========================================"
echo "TensorRT-LLM installed successfully!"
echo "========================================"
echo ""
echo "Next steps - build Whisper small.en engine:"
echo "  cd src/whisper_trtllm && ./build_small_en.sh"
echo ""
echo "Memory tips (from README4Jetson.md):"
echo "  - Use --use_mmap flag to reduce memory"
echo "  - With CUDA_LAZY_LOADING: ~6.8GB GPU, ~7.3GB total"
echo ""
