"""Local Whisper transcription — timestamp-aware, fully offline."""
from __future__ import annotations
import os
from typing import Iterator

_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


class WhisperTranscriber:
    def __init__(self, model_size: str = "base"):
        from faster_whisper import WhisperModel
        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=_MODEL_DIR,
        )
        self._model_size = model_size
        print(f"[whisper] Loaded '{model_size}' model")

    def transcribe_iter(self, audio_path: str) -> Iterator[dict]:
        """Stream transcript segments with start/end timestamps."""
        segments, info = self._model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=False,
            language=None,   # auto-detect
        )
        for seg in segments:
            text = seg.text.strip()
            if text:
                yield {
                    "text": text,
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                }

    def transcribe(self, audio_path: str) -> list[dict]:
        return list(self.transcribe_iter(audio_path))
