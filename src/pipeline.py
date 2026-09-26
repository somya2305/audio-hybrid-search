"""End-to-end pipeline: audio -> transcript -> chunks -> embeddings -> PostgreSQL.

For each audio file: transcribe + diarize with WhisperX (writes
output/transcription/<stem>_transcription.json), then chunk, embed and index
it in one transaction (see src/index_chunks.py).

Transcription is the slow step (~12 min per 9-min file on CPU), so a file
whose transcript already exists is not re-transcribed unless --force is given.

Usage:
    python -m src.pipeline                       # every file in data/audio/
    python -m src.pipeline data/audio/health_01.wav
    python -m src.pipeline --force               # re-transcribe everything
"""

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:  # also allow `python src/pipeline.py`
    sys.path.insert(0, str(ROOT_DIR))

from src.database.connection import get_connection
from src.database.repository import count_chunks
from src.diarization.diarize import OUTPUT_DIR as TRANSCRIPTION_DIR
from src.diarization.diarize import diarize, find_audio_files
from src.index_chunks import index_transcription_file


def transcription_path_for(audio_path):
    return TRANSCRIPTION_DIR / f"{Path(audio_path).stem}_transcription.json"


def process_audio(conn, audio_path, force=False, num_speakers=2):
    """Transcribe (unless a transcript exists) and index one audio file."""
    audio_path = Path(audio_path)
    transcription_path = transcription_path_for(audio_path)

    print(f"\n=== {audio_path.name} ===")
    if transcription_path.exists() and not force:
        print(f"Transcript exists, skipping transcription ({transcription_path.relative_to(ROOT_DIR)})")
    else:
        diarize(audio_path, transcription_path, num_speakers)

    return index_transcription_file(conn, transcription_path)


def main():
    parser = argparse.ArgumentParser(description="Transcribe, chunk, embed and index audio files.")
    parser.add_argument(
        "audio_paths", nargs="*",
        help="Audio files to process (default: every audio file in data/audio/)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-transcribe even when a transcript already exists",
    )
    parser.add_argument(
        "--num-speakers", type=int, default=2, help="Number of speakers (default: 2)"
    )
    args = parser.parse_args()

    audio_files = [Path(p) for p in args.audio_paths] or find_audio_files()
    missing = [str(p) for p in audio_files if not p.is_file()]
    if missing:
        sys.exit(f"Audio file(s) not found: {', '.join(missing)}")
    if not audio_files:
        sys.exit("No audio files found in data/audio/")

    total_start = time.perf_counter()
    results, failed = [], []
    print(f"Processing {len(audio_files)} audio file(s)")

    # autocommit=True so each file's conn.transaction() is a real commit.
    with get_connection(autocommit=True) as conn:
        for audio_file in audio_files:
            try:
                result = process_audio(conn, audio_file, args.force, args.num_speakers)
                if result:
                    results.append(result)
            except Exception as e:  # keep going; one bad file shouldn't stop the rest
                failed.append(audio_file.name)
                print(f"{audio_file.name}: ERROR {type(e).__name__}: {e}")

        conversations = conn.execute("SELECT count(*) FROM conversations").fetchone()[0]
        db_chunks = count_chunks(conn)

    print(
        f"\nProcessed {len(results)}/{len(audio_files)} files, "
        f"{sum(r['num_chunks'] for r in results)} chunks in "
        f"{time.perf_counter() - total_start:.1f}s"
    )
    print(f"Database: {conversations} conversations, {db_chunks} chunks")
    if failed:
        sys.exit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
