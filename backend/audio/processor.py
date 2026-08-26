"""Audio metadata extraction — duration, format info."""
from __future__ import annotations
import json
import os
import subprocess


def get_audio_meta(path: str, filename: str) -> dict:
    return {
        "filename": filename,
        "duration": _duration(path),
        "size_mb": round(os.path.getsize(path) / 1_048_576, 2),
    }


def _duration(path: str) -> float:
    # Primary: ffprobe
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", path],
            capture_output=True, text=True, timeout=15,
        )
        data = json.loads(out.stdout)
        for stream in data.get("streams", []):
            d = stream.get("duration")
            if d:
                return float(d)
    except Exception:
        pass

    # Fallback: mutagen
    try:
        from mutagen import File as MFile
        audio = MFile(path)
        if audio and hasattr(audio, "info") and audio.info:
            return float(audio.info.length)
    except Exception:
        pass

    return 0.0
