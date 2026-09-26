"""End-to-end pipeline: audio -> transcript -> chunks -> embeddings -> PostgreSQL.

Transcription is the slow step (~12 min per 9-min file on CPU), so audio that
already has a transcript in output/transcription/ is not re-transcribed
unless --force is given.

Usage:
    python -m src.pipeline                       # every file in data/audio/
    python -m src.pipeline data/audio/health_01.wav
    python -m src.pipeline --force               # re-transcribe everything
"""

import argparse
import sys
from pathlib import Path

from src.common import TRANSCRIPTION_DIR
from src.diarization.diarize import diarize, find_audio_files
from src.index_chunks import index_files


def main():
    parser = argparse.ArgumentParser(description="Transcribe, chunk, embed and index audio files.")
    parser.add_argument(
        "audio_paths", nargs="*", help="Audio files (default: every audio file in data/audio/)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-transcribe even if a transcript exists"
    )
    parser.add_argument("--num-speakers", type=int, default=2, help="Number of speakers (default: 2)")
    args = parser.parse_args()

    audio_files = [Path(p) for p in args.audio_paths] or find_audio_files()
    missing = [str(p) for p in audio_files if not p.is_file()]
    if missing or not audio_files:
        sys.exit(f"Audio file(s) not found: {', '.join(missing) or 'data/audio/ is empty'}")

    transcripts, failed = [], []
    for audio_file in audio_files:
        transcript = TRANSCRIPTION_DIR / f"{audio_file.stem}_transcription.json"
        if args.force or not transcript.exists():
            print(f"Transcribing {audio_file.name}")
            try:
                diarize(audio_file, transcript, args.num_speakers)
            except Exception as e:
                failed.append(audio_file.name)
                print(f"{audio_file.name}: ERROR {type(e).__name__}: {e}")
                continue
        transcripts.append(transcript)

    if transcripts:
        failed += index_files(transcripts)
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
