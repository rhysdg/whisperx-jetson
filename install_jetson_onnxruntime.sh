#!/bin/bash
# Install ONNX Runtime for Jetson JetPack 6, Python 3.10
# From: https://www.elinux.org/Jetson_Zoo#ONNX_Runtime

set -e

# ONNX Runtime 1.19.0 for JetPack 6 (L4T R36.x), Python 3.10
WHEEL_URL="https://nvidia.box.com/shared/static/48dtuob7meiw6ebgfsfqakc9vse62sg4.whl"
WHEEL_NAME="onnxruntime_gpu-1.19.0-cp310-cp310-linux_aarch64.whl"

echo "=== Installing ONNX Runtime 1.19.0 for JetPack 6 (Python 3.10) ==="
echo ""

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
if [ "$PYTHON_VERSION" != "3.10" ]; then
    echo "Warning: This wheel is for Python 3.10, but you have Python $PYTHON_VERSION"
    echo "The installation may fail or not work correctly."
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Downgrade numpy first (Jetson onnxruntime requires numpy<2)
echo "Downgrading NumPy to <2.0 for compatibility..."
pip install "numpy>=1.24.0,<2.0.0"

# Download the wheel
echo "Downloading ONNX Runtime wheel..."
wget -O "$WHEEL_NAME" "$WHEEL_URL"

# Uninstall existing onnxruntime if present
echo "Removing any existing onnxruntime installations..."
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true

# Install the wheel
echo "Installing ONNX Runtime GPU wheel..."
pip install "$WHEEL_NAME"

# Clean up
rm -f "$WHEEL_NAME"

# Verify installation
echo ""
echo "=== Verifying installation ==="
python3 -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')"
python3 -c "import onnxruntime; print(f'Available providers: {onnxruntime.get_available_providers()}')"

echo ""
echo "=== Installation complete! ==="
