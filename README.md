# WhisperX for Jetson

**Jetson-optimized fork of [WhisperX](https://github.com/m-bain/whisperX) by Max Bain**

Fast automatic speech recognition with word-level timestamps and speaker diarization on NVIDIA Jetson devices.

## Tested On

| Device | JetPack |
|--------|---------|
| Orin Nano Super | 6.2.0 |

## Jetson Installation

### Requirements

- NVIDIA Jetson (Orin Nano, AGX Orin, etc.)
- JetPack 6 (L4T R36.x)
- Python 3.10

### Install

```bash
git clone https://github.com/disler/whisperx-jetson.git
cd whisperx-jetson
chmod +x install_jetson.sh
./install_jetson.sh
```

### Enable GPU Acceleration (Optional)

The pip version of CTranslate2 doesn't include CUDA support for aarch64. To enable GPU acceleration, build from source:

```bash
chmod +x build_ctranslate2_cuda.sh
./build_ctranslate2_cuda.sh
```

This takes ~20-30 minutes but enables CUDA inference. Without it, WhisperX falls back to CPU.

### Verify

```bash
python3 -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
python3 -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
python3 -c "import onnxruntime; print(f'Providers: {onnxruntime.get_available_providers()}')"
```

You should see `CUDAExecutionProvider` and `TensorrtExecutionProvider` in the providers list.

### Key Dependencies

PyTorch is installed from [Jetson AI Lab PyPI](https://pypi.jetson-ai-lab.io/jp6/cu126) which has wheels built for JetPack 6 with cuDNN 9 support.

## Usage

### File Transcription

```bash
# int8 compute type required on Jetson (float16 not supported)
whisperx audio.wav --compute_type int8
```

### Realtime Streaming (Mic to Text)

Stream microphone audio to text in realtime. Primary target: **Seeed Studio ReSpeaker Mic Array** (adaptable to other mics).

#### Install PyAudio (required for realtime)

```bash
sudo apt-get install -y portaudio19-dev python3-pyaudio
pip install pyaudio
```

#### Model Selection (Memory Considerations)

Orin Nano (8GB) memory limits:

| Model | Memory | Realtime |
|-------|--------|----------|
| `tiny` / `tiny.en` | ~1GB | ✅ Recommended |
| `base` / `base.en` | ~1.5GB | ✅ Works |
| `small` / `small.en` | ~2.5GB | ⚠️ May OOM |
| `medium` / `large` | 5GB+ | ❌ Won't fit |

For better accuracy within memory limits, use `base.en` (English-only, optimized).

#### CLI Usage

```bash
# List available microphones
python -m whisperx.realtime --list-devices

# Start realtime transcription (auto-detects ReSpeaker or default mic)
python -m whisperx.realtime --model tiny --compute-type int8

# Use base.en for better accuracy (still fits in 8GB)
python -m whisperx.realtime --model base.en --compute-type int8

# Specify microphone by device index
python -m whisperx.realtime --model tiny --compute-type int8 --mic-device 0

# Output as JSON (for piping to LLM)
python -m whisperx.realtime --json | your_llm_script.py
```

#### Python API

```python
from whisperx.realtime import RealtimeTranscriber

transcriber = RealtimeTranscriber(
    model_name="base",
    compute_type="int8",  # Required on Jetson
    device_index=None,    # Auto-detects ReSpeaker, or specify index
)

# Stream transcripts
for text in transcriber.stream():
    print(text)  # Send to LLM, CLI, etc.
```

## Roadmap

- [x] Realtime audio chunk-based processing for streaming transcription
- [ ] TensorRT optimization for faster inference
- [ ] VAD-based chunking improvements
- [ ] Multi-language realtime support

## Attribution

Based on [WhisperX](https://github.com/m-bain/whisperX) by Max Bain et al. See [LICENSE](LICENSE) for details.