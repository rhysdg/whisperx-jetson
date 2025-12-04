#!/bin/bash
# Full installation script for WhisperX on Jetson JetPack 6
# Handles onnxruntime and numpy compatibility automatically

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== WhisperX Jetson Installation ==="
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

# Step 1: Thoroughly remove ALL existing onnxruntime installations
echo ""
echo "=== Step 1: Removing ALL existing ONNX Runtime installations ==="
pip uninstall -y onnxruntime 2>/dev/null || true
pip uninstall -y onnxruntime-gpu 2>/dev/null || true
pip uninstall -y onnxruntime-silicon 2>/dev/null || true
pip uninstall -y onnxruntime-directml 2>/dev/null || true
pip uninstall -y ort-nightly 2>/dev/null || true
pip uninstall -y ort-nightly-gpu 2>/dev/null || true

# Force remove any remaining onnxruntime files from site-packages
SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "Cleaning up any remaining onnxruntime files in $SITE_PACKAGES..."
rm -rf "$SITE_PACKAGES/onnxruntime"* 2>/dev/null || true

# Verify removal
if python3 -c "import onnxruntime" 2>/dev/null; then
    echo "ERROR: onnxruntime is still installed! Please manually remove it."
    python3 -c "import onnxruntime; print(f'Found version: {onnxruntime.__version__}')"
    exit 1
else
    echo "ONNX Runtime successfully removed."
fi

# Step 2: Install numpy<2 (required for Jetson onnxruntime)
echo ""
echo "=== Step 2: Installing NumPy <2.0 ==="
pip install "numpy>=1.24.0,<2.0.0"

# Step 3: Install Jetson ONNX Runtime GPU
echo ""
echo "=== Step 3: Installing ONNX Runtime GPU 1.19.0 for Jetson ==="
WHEEL_URL="https://nvidia.box.com/shared/static/48dtuob7meiw6ebgfsfqakc9vse62sg4.whl"
WHEEL_NAME="onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl"

# Download and install
wget -O "$WHEEL_NAME" "$WHEEL_URL"
pip install --no-deps "$WHEEL_NAME"
rm -f "$WHEEL_NAME"

# Verify onnxruntime version is correct
echo ""
echo "Verifying ONNX Runtime installation..."
INSTALLED_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)")
# Accept 1.18.0 or 1.19.0 (NVIDIA wheel reports 1.18.0 despite filename)
if [[ "$INSTALLED_VERSION" != "1.18.0" && "$INSTALLED_VERSION" != "1.19.0" ]]; then
    echo "ERROR: Expected onnxruntime 1.18.0 or 1.19.0 but got $INSTALLED_VERSION"
    exit 1
fi
echo "ONNX Runtime version: $INSTALLED_VERSION ✓"
python3 -c "import onnxruntime; print(f'Providers: {onnxruntime.get_available_providers()}')"

# Step 4: Install WhisperX in editable mode (skips conflicting deps on aarch64)
echo ""
echo "=== Step 4: Installing WhisperX ==="
cd "$SCRIPT_DIR"
pip install --no-build-isolation --no-deps -e .

# Install remaining dependencies (excluding onnxruntime and numpy)
# Install faster-whisper with --no-deps to prevent it from pulling onnxruntime
echo ""
echo "=== Step 5: Installing remaining dependencies ==="
pip install --no-deps faster-whisper
pip install ctranslate2 nltk pandas av pyannote-audio transformers huggingface-hub tokenizers

# Step 6: Verify onnxruntime wasn't overwritten
echo ""
echo "=== Step 6: Verifying ONNX Runtime wasn't overwritten ==="
FINAL_VERSION=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)")
if [[ "$FINAL_VERSION" != "1.18.0" && "$FINAL_VERSION" != "1.19.0" ]]; then
    echo "WARNING: onnxruntime was overwritten! Got $FINAL_VERSION instead of 1.18.0"
    echo "Reinstalling Jetson ONNX Runtime..."
    pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true
    SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
    rm -rf "$SITE_PACKAGES/onnxruntime"* 2>/dev/null || true
    wget -O "$WHEEL_NAME" "$WHEEL_URL"
    pip install --no-deps "$WHEEL_NAME"
    rm -f "$WHEEL_NAME"
fi

# Step 7: Final verification
echo ""
echo "=== Final Verification ==="
python3 -c "import whisperx; print('WhisperX imported successfully!')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
python3 -c "import onnxruntime; print(f'Providers: {onnxruntime.get_available_providers()}')"

# Check for GPU providers
if python3 -c "import onnxruntime; assert 'CUDAExecutionProvider' in onnxruntime.get_available_providers()" 2>/dev/null; then
    echo "✓ CUDA provider available"
else
    echo "✗ WARNING: CUDA provider NOT available - check your installation"
fi

echo ""
echo "=== Installation Complete! ==="
echo ""
echo "Usage: whisperx audio.wav --model large-v3 --compute_type float16"
