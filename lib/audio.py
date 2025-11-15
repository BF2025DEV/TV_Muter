"""
Audio capture and playback utilities for TV Muter
"""

import pyaudio
import numpy as np
import wave
import soundfile as sf
from typing import Optional, List, Tuple


class AudioCapture:
    """Handle audio input from microphone or line-in"""

    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 device_index: Optional[int] = None):
        """
        Initialize audio capture

        Args:
            sample_rate: Sampling rate in Hz (16000 is good for speech/commercial detection)
            channels: Number of audio channels (1=mono, 2=stereo)
            chunk_size: Number of samples per buffer
            device_index: Which audio device to use (None = default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device_index = device_index

        self.p = pyaudio.PyAudio()
        self.stream = None

    def list_devices(self) -> List[dict]:
        """List all available audio input devices"""
        devices = []
        for i in range(self.p.get_device_count()):
            dev_info = self.p.get_device_info_by_index(i)
            if dev_info['maxInputChannels'] > 0:  # Only input devices
                devices.append({
                    'index': i,
                    'name': dev_info['name'],
                    'channels': dev_info['maxInputChannels'],
                    'sample_rate': int(dev_info['defaultSampleRate'])
                })
        return devices

    def start(self):
        """Start capturing audio"""
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=self.chunk_size
        )
        print(f"Audio capture started (SR: {self.sample_rate}Hz, Channels: {self.channels})")

    def read_chunk(self) -> np.ndarray:
        """
        Read one chunk of audio data

        Returns:
            numpy array of audio samples (normalized to -1.0 to 1.0)
        """
        if self.stream is None:
            raise RuntimeError("Audio stream not started. Call start() first.")

        data = self.stream.read(self.chunk_size, exception_on_overflow=False)
        # Convert bytes to numpy array and normalize
        audio_data = np.frombuffer(data, dtype=np.int16)
        return audio_data.astype(np.float32) / 32768.0

    def stop(self):
        """Stop capturing audio"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        print("Audio capture stopped")

    def close(self):
        """Clean up resources"""
        self.stop()
        self.p.terminate()

    def __enter__(self):
        """Context manager support"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()


class AudioRecorder:
    """Record audio to file with real-time monitoring"""

    def __init__(self, capture: AudioCapture):
        """
        Initialize recorder

        Args:
            capture: AudioCapture instance to record from
        """
        self.capture = capture
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        """Start recording audio"""
        self.frames = []
        self.is_recording = True
        print("Recording started...")

    def add_chunk(self, audio_chunk: np.ndarray):
        """Add audio chunk to recording"""
        if self.is_recording:
            self.frames.append(audio_chunk)

    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        print("Recording stopped")

    def save(self, filename: str):
        """
        Save recording to WAV file

        Args:
            filename: Path to save file
        """
        if not self.frames:
            print("No audio to save")
            return

        # Concatenate all chunks
        audio_data = np.concatenate(self.frames)

        # Save as WAV file using soundfile (handles normalization)
        sf.write(filename, audio_data, self.capture.sample_rate)

        duration = len(audio_data) / self.capture.sample_rate
        print(f"Saved {duration:.1f} seconds of audio to {filename}")

    def get_level(self) -> float:
        """
        Get current audio level (RMS)

        Returns:
            Audio level from 0.0 to 1.0
        """
        if not self.frames:
            return 0.0

        # Use last chunk for real-time level
        recent_audio = self.frames[-1]
        rms = np.sqrt(np.mean(recent_audio ** 2))
        return float(rms)


def load_audio(filename: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    Load audio file

    Args:
        filename: Path to audio file
        target_sr: Target sample rate (will resample if needed)

    Returns:
        Tuple of (audio_data, sample_rate)
    """
    import librosa
    audio, sr = librosa.load(filename, sr=target_sr, mono=True)
    return audio, sr


def get_audio_level(audio_data: np.ndarray) -> float:
    """
    Calculate RMS level of audio

    Args:
        audio_data: Audio samples

    Returns:
        RMS level (0.0 to 1.0)
    """
    rms = np.sqrt(np.mean(audio_data ** 2))
    return float(rms)
