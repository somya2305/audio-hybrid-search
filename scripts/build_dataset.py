"""Build the golden audio dataset from data/dataset.json.

For every clip in the manifest this downloads the source (YouTube / podcast
URL via yt-dlp, or a local file), cuts the requested section, converts it to
16 kHz mono WAV in data/audio/, and checks the result is 8-10 minutes long.

Usage:
    python scripts/build_dataset.py                  # build every clip
    python scripts/build_dataset.py diary_ceo_01.wav # build only these
    python scripts/build_dataset.py --force          # rebuild existing clips

Manifest entry (data/dataset.json -> "clips"):
    {
      "file": "diary_ceo_01.wav",        # output name in data/audio/
      "source": "https://...",           # URL, or a local path
      "clip_start": "00:12:00",          # HH:MM:SS, MM:SS or seconds
      "duration_s": 540,                 # optional, default 540 (9 min)
      "show": "...", "speakers": ["...", "..."], "topic": "..."
    }
Entries whose source is empty or starts with "TODO" are skipped.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

MANIFEST_PATH = ROOT_DIR / "data" / "dataset.json"
AUDIO_DIR = ROOT_DIR / "data" / "audio"
RAW_DIR = ROOT_DIR / "data" / "raw"

DEFAULT_DURATION_S = 540
MIN_DURATION_S = 8 * 60
MAX_DURATION_S = 10 * 60
SAMPLE_RATE = 16000


def parse_time(value):
    """Seconds from "HH:MM:SS", "MM:SS" or a number."""
    if isinstance(value, (int, float)):
        return float(value)

    seconds = 0.0
    for part in str(value).split(":"):
        seconds = seconds * 60 + float(part)
    return seconds


def is_url(source):
    return source.startswith(("http://", "https://"))


def probe_duration(path):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(result.stdout.strip())


def fetch_source(clip):
    """Return a local path to the clip's full source audio."""
    source = clip["source"]

    if not is_url(source):
        path = Path(source)
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.exists():
            raise FileNotFoundError(f"Local source not found: {path}")
        return path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(clip["file"]).stem

    # Downloads are cached so re-trimming doesn't re-download.
    cached = sorted(RAW_DIR.glob(f"{stem}.*"))
    if cached:
        print(f"  using cached download: {cached[0].name}")
        return cached[0]

    print(f"  downloading: {source}")
    subprocess.run(
        [
            sys.executable, "-m", "yt_dlp",
            "--no-playlist",
            "-x", "--audio-format", "mp3",
            "-o", str(RAW_DIR / f"{stem}.%(ext)s"),
            source,
        ],
        check=True,
    )

    downloaded = sorted(RAW_DIR.glob(f"{stem}.*"))
    if not downloaded:
        raise RuntimeError("yt-dlp finished but no file was written")
    return downloaded[0]


def trim(source_path, output_path, start_s, duration_s):
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-ss", str(start_s),
            "-t", str(duration_s),
            "-i", str(source_path),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            str(output_path),
        ],
        check=True,
    )


def build_clip(clip, force=False):
    output_path = AUDIO_DIR / clip["file"]

    if output_path.exists() and not force:
        return "exists", probe_duration(output_path)

    start_s = parse_time(clip.get("clip_start", 0))
    duration_s = float(clip.get("duration_s", DEFAULT_DURATION_S))

    source_path = fetch_source(clip)

    source_duration = probe_duration(source_path)
    if start_s + duration_s > source_duration:
        raise ValueError(
            f"clip {start_s:.0f}s + {duration_s:.0f}s runs past the end "
            f"of the source ({source_duration:.0f}s)"
        )

    print(f"  trimming {start_s:.0f}s -> {start_s + duration_s:.0f}s")
    trim(source_path, output_path, start_s, duration_s)

    return "built", probe_duration(output_path)


def check_manifest(clips):
    """Print warnings for duplicate files or speakers reused across clips."""
    warnings = []

    files = [clip["file"] for clip in clips]
    for name in sorted({f for f in files if files.count(f) > 1}):
        warnings.append(f"file name used more than once: {name}")

    seen = {}
    for clip in clips:
        for speaker in clip.get("speakers", []):
            key = speaker.strip().lower()
            if key and not key.startswith("todo"):
                seen.setdefault(key, []).append(clip["file"])
    for speaker, clip_files in seen.items():
        if len(clip_files) > 1:
            warnings.append(
                f"speaker '{speaker}' appears in {', '.join(clip_files)} "
                f"(each clip should have a unique pair)"
            )

    for clip in clips:
        if len(clip.get("speakers", [])) != 2:
            warnings.append(f"{clip['file']}: expected exactly 2 speakers")

    return warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("files", nargs="*", help="only build these clips")
    parser.add_argument("--force", action="store_true", help="rebuild existing clips")
    args = parser.parse_args()

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        clips = json.load(f)["clips"]

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    selected = [c for c in clips if not args.files or c["file"] in args.files]
    results = []

    for clip in selected:
        name = clip["file"]
        source = (clip.get("source") or "").strip()

        print(f"\n{name}")

        if not source or source.upper().startswith("TODO"):
            print("  skipped: no source yet")
            results.append((name, "no source", None))
            continue

        try:
            status, duration = build_clip(clip, force=args.force)
            if not MIN_DURATION_S <= duration <= MAX_DURATION_S:
                status += " (length outside 8-10 min)"
            results.append((name, status, duration))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((name, "error", None))

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for name, status, duration in results:
        length = f"{duration / 60:5.2f} min" if duration else "    -    "
        print(f"  {name:32s} {length}  {status}")

    ready = sum(
        1 for _, status, _ in results
        if status in ("built", "exists")
    )
    print(f"\n{ready}/{len(results)} clips ready in {os.path.relpath(AUDIO_DIR, ROOT_DIR)}/")

    for warning in check_manifest(clips):
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()
