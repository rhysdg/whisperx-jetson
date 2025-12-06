# WhisperJet

**High-performance realtime speech recognition for NVIDIA Jetson**

WhisperJet provides fast automatic speech recognition on Jetson devices with **dual-backend support**:

| Backend | Library | Quantization | Use Case |
|---------|---------|--------------|----------|
| **CTranslate2** | [WhisperX](https://github.com/m-bain/whisperX) | int8 | Word-level timestamps, speaker diarization |
| **TensorRT-LLM** | NVIDIA TRT-LLM 0.12 | int8 | Fastest inference, lowest latency |

> **Credits:** This project builds on the excellent work of [Max Bain's WhisperX](https://github.com/m-bain/whisperX) and NVIDIA's TensorRT-LLM Whisper examples - all adapted to Jetson specific dependencies and constraints.

## Tested On

| Device | JetPack |
|--------|---------|
| Orin Nano Super | 6.2.0 |

## Installation

### Requirements

- NVIDIA Jetson (Orin Nano, AGX Orin, etc.)
- JetPack 6 (L4T R36.x)
- Python 3.10

### Install WhisperJet

```bash
git clone https://github.com/rhysdg/mr-b.git
cd mr-b/whisperx-jetson
chmod +x install_jetson.sh
./install_jetson.sh
```

### Enable CTranslate2 GPU Acceleration (Optional)

The pip version of CTranslate2 doesn't include CUDA support for aarch64. To enable GPU acceleration for the WhisperX backend, build from source:

```bash
chmod +x build_ctranslate2_cuda.sh
./build_ctranslate2_cuda.sh
```

This takes ~20-30 minutes but enables CUDA inference. Without it, the CTranslate2 backend falls back to CPU.

### Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}')"
python3 -c "import tensorrt_llm; print(f'TensorRT-LLM: {tensorrt_llm.__version__}')"
```

### Key Dependencies

PyTorch is installed from [Jetson AI Lab PyPI](https://pypi.jetson-ai-lab.io/jp6/cu126) which has wheels built for JetPack 6 with cuDNN 9 support.

## Realtime Transcription

Stream microphone audio to text in realtime. Primary target: **Seeed Studio ReSpeaker Mic Array** (adaptable to other mics).

### Install PyAudio (required for realtime)

```bash
sudo apt-get install -y portaudio19-dev python3-pyaudio
pip install pyaudio
```

### Backend Comparison

| Backend | Latency | Memory | Features |
|---------|---------|--------|----------|
| **TensorRT-LLM** | ~100ms | ~2GB | Fastest, int8 engine |
| **CTranslate2** | ~200ms | ~1-2.5GB | Timestamps, diarization |

### CLI Usage

```bash
# List available microphones
python -m whisperjet.realtime --list-devices

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TensorRT-LLM Backend (Recommended for lowest latency)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
python -m whisperjet.realtime \
    --backend tensorrt-llm \
    --engine_dir whisper_trtllm/whisper_small.en_int8 \
    --max_new_tokens 48

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CTranslate2 Backend (Word timestamps & diarization)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Start realtime transcription (auto-detects ReSpeaker or default mic)
python -m whisperjet.realtime --model tiny --compute-type int8

# Use base.en for better accuracy (still fits in 8GB)
python -m whisperjet.realtime --model base.en --compute-type int8

# Specify microphone by device index
python -m whisperjet.realtime --model tiny --device_index 2
```

### Model Selection (Memory Considerations)

#### CTranslate2 Backend

Orin Nano (8GB) memory limits:

| Model | Memory | Realtime |
|-------|--------|----------|
| `tiny` / `tiny.en` | ~1GB | Recommended |
| `base` / `base.en` | ~1.5GB | Works |
| `small` / `small.en` | ~2.5GB | May OOM |
| `medium` / `large` | 5GB+ | Won't fit |

#### TensorRT-LLM Backend (int8)

| Model | Memory | Realtime | Status |
|-------|--------|----------|--------|
| `tiny.en` | ~1GB | Fast | Tested |
| `base.en` | ~1.5GB | Fast | Tested |
| `small.en` | ~2GB | Fast | Tested |
| `medium.en` | ~3GB | Unknown | Not tested |
| `large-v3` | ~5GB | May OOM | Not tested |

> **Note:** TensorRT-LLM int8 engines are more memory-efficient than CTranslate2, allowing `small.en` to run comfortably on 8GB Orin Nano.

## Building TensorRT-LLM Engines

See `whisper_trtllm/README.md` for instructions on building int8 Whisper engines.

Quick start:
```bash
cd whisper_trtllm
./build_small_en.sh
```

## File Transcription

```bash
# CTranslate2 backend
whisperx audio.wav --compute_type int8

# TensorRT-LLM backend  
cd whisper_trtllm
python3 run.py --engine_dir whisper_small.en_int8 --input_file audio.wav --name small.en
```

## Patch Notes

- Added dual-backend support: CTranslate2 (WhisperX) and TensorRT-LLM
- TensorRT-LLM 0.12 integration with int8 quantization
- Realtime CLI with `--backend` flag to switch between engines
- Fixed PyAudio threading issues on Jetson (blocking mode for ARM compatibility)
- Added `--max_new_tokens`, `--max_input_len`, `--padding_strategy` CLI flags for TensorRT-LLM tuning
