#!/bin/bash
# WhisperJet CTranslate2 Backend Installation for Jetson JetPack 6
# 
# This script installs the CTranslate2/WhisperX backend dependencies.
# Run AFTER setup_tensorrt_llm.sh and setup_venv.sh which handle:
#   - PyTorch (from Jetson AI Lab)
#   - numpy<2
#   - TensorRT-LLM
#
# Installation order:
#   1. setup_tensorrt_llm.sh  - Build TensorRT-LLM + install torch
#   2. setup_venv.sh          - Create virtualenv with CUDA symlinks
#   3. install_jetson.sh      - Install CTranslate2 backend (this script)
#   4. build_ctranslate2_cuda.sh - (Optional) Build CTranslate2 with CUDA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "WhisperJet CTranslate2 Backend Installation"
echo "=========================================="
echo ""

# Check if we're on aarch64 (Jetson)
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "Warning: This script is designed for Jetson (aarch64), detected: $ARCH"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $PYTHON_VERSION"

if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "Warning: This script is optimized for Python 3.10, you have $PYTHON_VERSION"
fi

# Check prerequisites from setup_tensorrt_llm.sh / setup_venv.sh
echo ""
echo "[1/5] Checking prerequisites..."

# Check PyTorch
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: PyTorch with CUDA not found!"
    echo "Please run setup_tensorrt_llm.sh first to install PyTorch."
    exit 1
fi
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "  PyTorch: $TORCH_VERSION (CUDA OK)"

# Check numpy
if ! python3 -c "import numpy" 2>/dev/null; then
    echo "ERROR: numpy not found!"
    echo "Please run setup_venv.sh first."
    exit 1
fi
NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)")
echo "  NumPy: $NUMPY_VERSION"

# Step 2: Clean up any existing onnxruntime installations
echo ""
echo "[2/5] Cleaning existing ONNX Runtime installations..."
pip uninstall -y onnxruntime 2>/dev/null || true
pip uninstall -y onnxruntime-gpu 2>/dev/null || true
pip uninstall -y onnxruntime-silicon 2>/dev/null || true
pip uninstall -y onnxruntime-directml 2>/dev/null || true
pip uninstall -y ort-nightly 2>/dev/null || true
pip uninstall -y ort-nightly-gpu 2>/dev/null || true

# Force remove any remaining onnxruntime files from site-packages
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
rm -rf "$SITE_PACKAGES/onnxruntime"* 2>/dev/null || true
echo "  Cleaned."

# Step 3: Install Jetson ONNX Runtime GPU
echo ""
echo "[3/5] Installing ONNX Runtime GPU 1.19.0 for Jetson..."
WHEEL_URL="https://nvidia.box.com/shared/static/48dtuob7meiw6ebgfsfqakc9vse62sg4.whl"
WHEEL_NAME="onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl"

wget -q -O "$WHEEL_NAME" "$WHEEL_URL"
pip install --quiet --no-deps "$WHEEL_NAME"
rm -f "$WHEEL_NAME"

# Verify onnxruntime version
INSTALLED_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)")
# Accept 1.18.0 or 1.19.0 (NVIDIA wheel reports 1.18.0 despite filename)
if [[ "$INSTALLED_VERSION" != "1.18.0" && "$INSTALLED_VERSION" != "1.19.0" ]]; then
    echo "ERROR: Expected onnxruntime 1.18.0 or 1.19.0 but got $INSTALLED_VERSION"
    exit 1
fi
echo "  ONNX Runtime: $INSTALLED_VERSION"

# Step 4: Install WhisperJet in editable mode
echo ""
echo "[4/5] Installing WhisperJet and dependencies..."
cd "$SCRIPT_DIR"

# Install whisperjet without deps (we manage them ourselves)
pip install --quiet --no-build-isolation --no-deps -e .

# Install faster-whisper with --no-deps to prevent it from pulling onnxruntime
pip install --quiet --no-deps faster-whisper

# Install remaining CTranslate2 backend dependencies
pip install --quiet ctranslate2 nltk pandas av pyannote-audio transformers huggingface-hub tokenizers

# Step 5: Verify onnxruntime wasn't overwritten by a dependency
echo ""
echo "[5/5] Verifying installation..."
FINAL_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)")
if [[ "$FINAL_VERSION" != "1.18.0" && "$FINAL_VERSION" != "1.19.0" ]]; then
    echo "WARNING: onnxruntime was overwritten! Got $FINAL_VERSION"
    echo "Reinstalling Jetson ONNX Runtime..."
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    rm -rf "$SITE_PACKAGES/onnxruntime"* 2>/dev/null || true
    wget -q -O "$WHEEL_NAME" "$WHEEL_URL"
    pip install --quiet --no-deps "$WHEEL_NAME"
    rm -f "$WHEEL_NAME"
fi

# Final verification
echo ""
python3 -c "import whisperjet; print('  WhisperJet: OK')"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})')"
python3 -c "import numpy; print(f'  NumPy: {numpy.__version__}')"
python3 -c "import onnxruntime; print(f'  ONNX Runtime: {onnxruntime.__version__}')"
python3 -c "import ctranslate2; print(f'  CTranslate2: {ctranslate2.__version__}')"

# Check for GPU providers
if python3 -c "import onnxruntime; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()" 2>/dev/null; then
    echo "  ONNX CUDA Provider: OK"
else
    echo "  ONNX CUDA Provider: NOT AVAILABLE (CPU fallback)"
fi

# Check CTranslate2 CUDA
if python3 -c "import ctranslate2; assert 'cuda' in ctranslate2.get_supported_compute_types('cuda')" 2>/dev/null; then
    echo "  CTranslate2 CUDA: OK"
else
    echo "  CTranslate2 CUDA: NOT AVAILABLE (run build_ctranslate2_cuda.sh to enable)"
fi

echo ""
echo "=========================================="
echo "CTranslate2 Backend Installation Complete!"
echo "=========================================="
echo ""
echo "Usage:"
echo "  # Realtime transcription"
echo "  python -m whisperjet.realtime --model tiny --compute-type int8"
echo ""
echo "  # File transcription"
echo "  whisperx audio.wav --model base.en --compute_type int8"
echo ""
echo "For CTranslate2 GPU acceleration, run:"
echo "  ./build_ctranslate2_cuda.sh"
echo ""
