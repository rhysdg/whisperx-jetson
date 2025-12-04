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

```bash
# int8 compute type required on Jetson (float16 not supported)
whisperx audio.wav --compute_type int8
```

## Roadmap

- [ ] Realtime audio chunk-based processing for streaming transcription

## Attribution

Based on [WhisperX](https://github.com/m-bain/whisperX) by Max Bain et al. See [LICENSE](LICENSE) for details.
