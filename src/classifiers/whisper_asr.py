"""
SmartHandover — Whisper ASR (Automatic Speech Recognition)
Wrapper around openai/whisper-small for audio-to-text transcription.
"""

import numpy as np
import whisper
import torch


class WhisperASR:
    """Whisper-based speech-to-text transcriber."""

    def __init__(self, model_size: str = "small", device: str = None):
        """Load the Whisper model.

        Args:
            model_size: One of 'tiny', 'base', 'small', 'medium', 'large'.
            device: 'cpu' or 'cuda'. Auto-detects if None.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = whisper.load_model(model_size, device=device)

    def transcribe(self, audio_array, sr: int = 16000) -> str:
        """Transcribe an audio waveform to text.

        Args:
            audio_array: 1-D NumPy array (float32, mono).
            sr: Sampling rate (Whisper expects 16000).

        Returns:
            Transcribed text string.
        """
        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.cpu().numpy()

        audio_array = audio_array.astype(np.float32)

        # Whisper expects a flat 1-D float32 array at 16kHz
        result = self.model.transcribe(audio_array, language="en", fp16=(self.device == "cuda"))
        return result["text"].strip()

    def transcribe_file(self, path: str) -> str:
        """Transcribe an audio file to text.

        Args:
            path: Path to audio file (wav, flac, mp3, etc.).

        Returns:
            Transcribed text string.
        """
        result = self.model.transcribe(path, language="en", fp16=(self.device == "cuda"))
        return result["text"].strip()
