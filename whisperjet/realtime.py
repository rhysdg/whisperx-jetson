"""
Realtime audio streaming transcription for WhisperJet.

Primary target: Seeed Studio ReSpeaker Mic Array
Adaptable to other microphones via device_index parameter.

Usage:
    # CLI
    python -m whisperjet.realtime
    
    # Python API
    from whisperjet.realtime import RealtimeTranscriber
    transcriber = RealtimeTranscriber()
    for text in transcriber.stream():
        print(text)
"""

import sys
import os
import time
import queue
import threading
import numpy as np
import pyaudio
import whisperjet
from typing import Iterator, Optional, Callable, Union

# Add whisper_trtllm to path for TensorRT-LLM backend
current_dir = os.path.dirname(os.path.abspath(__file__))
trt_llm_path = os.path.join(os.path.dirname(current_dir), "whisper_trtllm")
if trt_llm_path not in sys.path:
    sys.path.append(trt_llm_path)

class WhisperTRTLLMWrapper:
    """Wrapper for TensorRT-LLM Whisper model to match WhisperX interface."""
    def __init__(self, engine_dir: str, assets_dir: str = None, max_new_tokens: int = 48, max_input_len: int = 3000):
        try:
            from run import WhisperTRTLLM, align_mel_length
            from whisper_utils import log_mel_spectrogram
            from tensorrt_llm._utils import str_dtype_to_torch
        except ImportError:
            raise ImportError("Could not import WhisperTRTLLM. Ensure whisper_trtllm is in python path.")
            
        if assets_dir is None:
            assets_dir = os.path.join(trt_llm_path, "assets")
            
        self.model = WhisperTRTLLM(engine_dir, assets_dir=assets_dir)
        self.log_mel_spectrogram = log_mel_spectrogram
        self.align_mel_length = align_mel_length
        self.str_dtype_to_torch = str_dtype_to_torch
        self.max_new_tokens = max_new_tokens
        self.max_input_len = max_input_len
        self.n_mels = self.model.encoder.n_mels

    def transcribe(self, audio: np.ndarray, batch_size: int = 1, language: str = "en") -> dict:
        import re
        import torch
        
        # Convert to mel spectrogram
        # log_mel_spectrogram handles numpy array input
        mel = self.log_mel_spectrogram(audio, self.n_mels, device='cuda')
        
        # CRITICAL: Convert to float16 like the working example does
        mel = mel.type(self.str_dtype_to_torch('float16'))
        
        # Add batch dimension (1, n_mels, time)
        mel = mel.unsqueeze(0)
        
        # Repeat for batch size if needed (though usually 1 for realtime)
        if batch_size > 1:
            mel = mel.repeat(batch_size, 1, 1)
            
        # Align length (pad/trim to max_input_len frames as per TRT engine requirement)
        # Using 'max' strategy to ensure we match the engine profile
        mel = self.align_mel_length(mel, self.max_input_len, 'max')
        
        # Generate
        texts = self.model.process_batch(
            mel,
            text_prefix="<|startoftranscript|><|en|><|transcribe|><|notimestamps|>",
            max_new_tokens=self.max_new_tokens
        )
        
        # Strip special tokens like the working example does
        cleaned_texts = []
        for t in texts:
            cleaned = re.sub(r'<\|.*?\|>', '', t).strip()
            cleaned_texts.append(cleaned)
        
        # Format result
        return {'segments': [{'text': t} for t in cleaned_texts]}

class RealtimeTranscriber:
    """
    Realtime microphone to text streaming transcription.
    
    Optimized for Seeed Studio ReSpeaker Mic Array but works with any microphone.
    
    Args:
        model_name: Whisper model to use (default: "base")
        device: Device to run on ("cuda" or "cpu")
        compute_type: Compute type ("int8" required on Jetson, "float16" elsewhere)
        device_index: PyAudio device index (None = default mic)
        sample_rate: Audio sample rate (16000 for Whisper)
        chunk_duration: Duration of each audio chunk in seconds
        silence_threshold: RMS threshold for silence detection
        silence_duration: Seconds of silence before processing chunk
        channels: Number of audio channels (1 for mono)
        on_transcript: Optional callback for each transcript
    """
    
    def __init__(
        self,
        model_name: str = "base",
        device: str = "cuda",
        compute_type: str = "int8",
        device_index: Optional[int] = None,
        sample_rate: int = 16000,
        chunk_duration: float = 0.5,
        silence_threshold: float = 0.01,
        silence_duration: float = 0.8,
        channels: int = 1,
        batch_size: int = 1,
        on_transcript: Optional[Callable[[str], None]] = None,
        backend: str = "ctranslate2",
        trt_engine_dir: Optional[str] = None,
        trt_assets_dir: Optional[str] = None,
        max_new_tokens: int = 48,
        max_input_len: int = 3000,
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.channels = channels
        self.batch_size = batch_size
        self.on_transcript = on_transcript
        self.backend = backend
        self.trt_engine_dir = trt_engine_dir
        self.trt_assets_dir = trt_assets_dir
        self.max_new_tokens = max_new_tokens
        self.max_input_len = max_input_len
        
        self.chunk_size = int(sample_rate * chunk_duration)
        self.silence_chunks = int(silence_duration / chunk_duration)
        
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._model = None
        self._pyaudio = None
        self._stream = None
    
    def _load_model(self):
        """Load Whisper model."""
        if self._model is None:
            print(f"Loading model '{self.model_name}' using {self.backend} on {self.device}...")
            
            if self.backend == "tensorrt-llm":
                if not self.trt_engine_dir:
                    raise ValueError("trt_engine_dir must be provided for tensorrt-llm backend")
                self._model = WhisperTRTLLMWrapper(
                    self.trt_engine_dir, 
                    self.trt_assets_dir,
                    self.max_new_tokens,
                    self.max_input_len
                )
            else:
                try:
                    self._model = whisperjet.load_model(
                        self.model_name,
                        self.device,
                        compute_type=self.compute_type
                    )
                except ValueError as e:
                    if "CUDA support" in str(e):
                        print(f"[WARN] CTranslate2 not compiled with CUDA, falling back to CPU...")
                        self.device = "cpu"
                        self._model = whisperjet.load_model(
                            self.model_name,
                            self.device,
                            compute_type="int8"
                        )
                    else:
                        raise
            print(f"Model loaded on {self.device}.")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - runs in separate thread."""
        self._audio_queue.put(in_data)
        return (None, pyaudio.paContinue)
    
    def _get_rms(self, audio_chunk: np.ndarray) -> float:
        """Calculate RMS (volume) of audio chunk."""
        return np.sqrt(np.mean(audio_chunk ** 2))
    
    def list_devices(self) -> list[dict]:
        """List available audio input devices."""
        p = pyaudio.PyAudio()
        devices = []
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                    "sample_rate": int(info["defaultSampleRate"])
                })
        p.terminate()
        return devices
    
    def find_respeaker(self) -> Optional[int]:
        """Find Seeed Studio ReSpeaker device index."""
        devices = self.list_devices()
        for dev in devices:
            name_lower = dev["name"].lower()
            if "respeaker" in name_lower or "seeed" in name_lower:
                return dev["index"]
        return None
    
    def stream(self) -> Iterator[str]:
        """
        Stream transcriptions from microphone.
        
        Yields transcribed text segments as they become available.
        Uses voice activity detection to determine speech boundaries.
        
        Example:
            transcriber = RealtimeTranscriber()
            for text in transcriber.stream():
                print(text)  # Send to LLM, CLI, etc.
        """
        self._load_model()
        
        # Try to find ReSpeaker if no device specified
        if self.device_index is None:
            respeaker_idx = self.find_respeaker()
            if respeaker_idx is not None:
                print(f"Found ReSpeaker at device {respeaker_idx}")
                self.device_index = respeaker_idx
        
        self._pyaudio = pyaudio.PyAudio()
        
        # Get device info
        if self.device_index is not None:
            dev_info = self._pyaudio.get_device_info_by_index(self.device_index)
            print(f"Using device: {dev_info['name']}")
        else:
            print("Using default input device")
        
        # Use blocking (non-callback) mode to avoid SystemError on Jetson
        # The callback mode triggers threading issues with Python 3.10 on ARM
        self._stream = self._pyaudio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
        )
        
        self._running = True
        audio_buffer = []
        silent_chunks = 0
        is_speaking = False
        
        print("Listening... (Ctrl+C to stop)")
        
        try:
            while self._running:
                # Read audio data directly (blocking read, no callback/queue)
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
                
                # Convert bytes to numpy array
                chunk = np.frombuffer(data, dtype=np.float32)
                rms = self._get_rms(chunk)
                
                if rms > self.silence_threshold:
                    # Speech detected
                    if not is_speaking:
                        is_speaking = True
                        print("[listening]")
                    audio_buffer.append(chunk)
                    silent_chunks = 0
                elif is_speaking:
                    # Silence after speech
                    audio_buffer.append(chunk)
                    silent_chunks += 1
                    
                    if silent_chunks >= self.silence_chunks:
                        # End of utterance - process audio
                        if len(audio_buffer) > 0:
                            audio = np.concatenate(audio_buffer)
                            
                            # Transcribe
                            result = self._model.transcribe(
                                audio,
                                batch_size=self.batch_size,
                                language="en"  # Set to None for auto-detect
                            )
                            
                            # Extract text
                            text = " ".join([s["text"].strip() for s in result["segments"]])
                            
                            if text.strip():
                                if self.on_transcript:
                                    self.on_transcript(text)
                                yield text
                        
                        # Reset
                        audio_buffer = []
                        is_speaking = False
                        silent_chunks = 0
        
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the audio stream."""
        self._running = False
        
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pyaudio:
            self._pyaudio.terminate()


def main():
    """CLI entry point for realtime transcription."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Realtime microphone transcription with WhisperX"
    )
    parser.add_argument(
        "--model", "-m",
        default="base",
        help="Whisper model (tiny, base, small, medium, large-v3)"
    )
    parser.add_argument(
        "--device", "-d",
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--compute-type", "-c",
        default="int8",
        help="Compute type (int8 required on Jetson)"
    )
    parser.add_argument(
        "--mic-device", "-i",
        type=int,
        default=None,
        help="Microphone device index (default: auto-detect ReSpeaker or default)"
    )
    parser.add_argument(
        "--list-devices", "-l",
        action="store_true",
        help="List available audio input devices"
    )
    parser.add_argument(
        "--silence-threshold", "-t",
        type=float,
        default=0.01,
        help="RMS threshold for silence detection"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="Batch size for transcription (reduce for OOM errors)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON (for piping to LLM)"
    )
    parser.add_argument(
        "--backend",
        default="ctranslate2",
        choices=["ctranslate2", "tensorrt-llm"],
        help="Inference backend"
    )
    parser.add_argument(
        "--engine_dir",
        default=None,
        help="Path to TensorRT-LLM engine directory (required for tensorrt-llm backend)"
    )
    parser.add_argument(
        "--assets_dir",
        default=None,
        help="Path to TensorRT-LLM assets directory"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=48,
        help="Maximum new tokens for TensorRT-LLM generation"
    )
    
    args = parser.parse_args()
    
    # If listing devices, we don't need to load the model
    if args.list_devices:
        # Instantiate with minimal args just to access list_devices
        transcriber = RealtimeTranscriber(model_name="tiny", device="cpu")
        print("Available input devices:")
        for dev in transcriber.list_devices():
            print(f"  {dev['index']}: {dev['name']} ({dev['channels']}ch, {dev['sample_rate']}Hz)")
        return

    transcriber = RealtimeTranscriber(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        device_index=args.mic_device,
        silence_threshold=args.silence_threshold,
        batch_size=args.batch_size,
        backend=args.backend,
        trt_engine_dir=args.engine_dir,
        trt_assets_dir=args.assets_dir,
        max_new_tokens=args.max_new_tokens,
    )
    
    if args.json:
        import json
        for text in transcriber.stream():
            print(json.dumps({"transcript": text}), flush=True)
    else:
        for text in transcriber.stream():
            print(f"> {text}")


if __name__ == "__main__":
    main()
