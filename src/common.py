"""Project paths, .env loading and small shared helpers."""

from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
AUDIO_DIR = ROOT_DIR / "data" / "audio"
QUERIES_PATH = ROOT_DIR / "data" / "queries.json"
TRANSCRIPTION_DIR = ROOT_DIR / "output" / "transcription"
CHUNKS_DIR = ROOT_DIR / "output" / "chunks"
EVAL_RESULTS_PATH = ROOT_DIR / "output" / "eval" / "results.json"

# DB_* settings and HF_TOKEN; variables already set in the shell take precedence.
load_dotenv(ROOT_DIR / ".env")


def format_time(seconds):
    """83.46 -> "01:23.5"."""
    # Round first so 59.96 becomes "01:00.0", not "00:60.0".
    minutes, secs = divmod(round(float(seconds), 1), 60)
    return f"{int(minutes):02d}:{secs:04.1f}"


def transcription_paths(paths=None):
    """The given transcript paths, or every *_transcription.json in output/transcription/."""
    return [Path(p) for p in paths] if paths else sorted(TRANSCRIPTION_DIR.glob("*_transcription.json"))
