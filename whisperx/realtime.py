"""
Realtime audio streaming transcription for WhisperX.

Primary target: Seeed Studio ReSpeaker Mic Array
Adaptable to other microphones via device_index parameter.

Usage:
    # CLI
    python -m whisperx.realtime
    
    # Python API
    from whisperx.realtime import RealtimeTranscriber
    transcriber = RealtimeTranscriber()
    for text in transcriber.stream():
        print(text)
"""

import sys
import queue
import threading
import numpy as np
import pyaudio
import whisperx
from typing import Iterator, Optional, Callable


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
        on_transcript: Optional[Callable[[str], None]] = None,
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
        self.on_transcript = on_transcript
        
        self.chunk_size = int(sample_rate * chunk_duration)
        self.silence_chunks = int(silence_duration / chunk_duration)
        
        self._audio_queue: queue.Queue = queue.Queue()
        self._running = False
        self._model = None
        self._pyaudio = None
        self._stream = None
    
    def _load_model(self):
        """Load WhisperX model."""
        if self._model is None:
            print(f"Loading model '{self.model_name}' on {self.device}...")
            try:
                self._model = whisperx.load_model(
                    self.model_name,
                    self.device,
                    compute_type=self.compute_type
                )
            except ValueError as e:
                if "CUDA support" in str(e):
                    print(f"[WARN] CTranslate2 not compiled with CUDA, falling back to CPU...")
                    self.device = "cpu"
                    self._model = whisperx.load_model(
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
        
        self._stream = self._pyaudio.open(
            format=pyaudio.paFloat32,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback
        )
        
        self._running = True
        audio_buffer = []
        silent_chunks = 0
        is_speaking = False
        
        print("Listening... (Ctrl+C to stop)")
        
        try:
            while self._running:
                try:
                    data = self._audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
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
                                batch_size=1,
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
        "--json",
        action="store_true",
        help="Output as JSON (for piping to LLM)"
    )
    
    args = parser.parse_args()
    
    transcriber = RealtimeTranscriber(
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        device_index=args.mic_device,
        silence_threshold=args.silence_threshold,
    )
    
    if args.list_devices:
        print("Available input devices:")
        for dev in transcriber.list_devices():
            print(f"  {dev['index']}: {dev['name']} ({dev['channels']}ch, {dev['sample_rate']}Hz)")
        return
    
    if args.json:
        import json
        for text in transcriber.stream():
            print(json.dumps({"transcript": text}), flush=True)
    else:
        for text in transcriber.stream():
            print(f"> {text}")


if __name__ == "__main__":
    main()
