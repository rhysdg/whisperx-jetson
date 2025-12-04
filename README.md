# WhisperX for Jetson

**Jetson-optimized fork of [WhisperX](https://github.com/m-bain/whisperX) by Max Bain**

Fast automatic speech recognition with word-level timestamps and speaker diarization on NVIDIA Jetson devices.

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

### Verify

```bash
python3 -c "import onnxruntime; print(f'ONNX Runtime: {onnxruntime.__version__}')"
python3 -c "import onnxruntime; print(f'Providers: {onnxruntime.get_available_providers()}')"
```

You should see `CUDAExecutionProvider` and `TensorrtExecutionProvider` in the providers list.

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

#### CLI Usage

```bash
# List available microphones
python -m whisperx.realtime --list-devices

# Start realtime transcription (auto-detects ReSpeaker or default mic)
python -m whisperx.realtime --model tiny --compute-type int8

# Specify microphone by device index
python -m whisperx.realtime --model tiny --compute-type int8 --mic-device 7

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
- [ ] VAD-based chunking improvements
- [ ] Multi-language realtime support

## Attribution

Based on [WhisperX](https://github.com/m-bain/whisperX) by Max Bain et al. See [LICENSE](LICENSE) for details.