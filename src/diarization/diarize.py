import argparse
import json
import os
import sys
import time
import warnings
from functools import lru_cache
from pathlib import Path

# pyannote warns at import that torchcodec can't load FFmpeg; we pass it
# in-memory audio from whisperx.load_audio, so torchcodec is never used.
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIO_DIR = ROOT_DIR / "data" / "audio"
OUTPUT_DIR = ROOT_DIR / "output" / "transcription"
AUDIO_EXTENSIONS = {".wav", ".opus", ".mp3", ".m4a", ".flac", ".ogg"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_MODEL = "small"
LANGUAGE = "en"
BATCH_SIZE = 16


def _stage(name, start):
    print(f"  done: {name} ({time.perf_counter() - start:.1f}s)")


# Models are cached so processing multiple files loads each one only once.

@lru_cache(maxsize=1)
def get_transcription_model():
    return whisperx.load_model(
        WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE
    )


@lru_cache(maxsize=1)
def get_align_model():
    return whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)


@lru_cache(maxsize=1)
def get_diarization_model(hf_token):
    return DiarizationPipeline(token=hf_token, device=DEVICE)


def find_audio_files(audio_dir=AUDIO_DIR):
    """Return the audio files in audio_dir, sorted by name."""
    return sorted(
        p for p in Path(audio_dir).iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _clean_words(segment):
    """Return the segment's words, each with a start and end time.

    WhisperX's aligner leaves some tokens (e.g. numbers like "2024") without
    timestamps. A missing start begins where the previous word ended; a
    missing end runs until the next timed word starts (or the segment ends),
    with that gap shared evenly by consecutive untimed words.
    """
    raw = [w for w in segment.get("words", []) if w.get("word", "").strip()]
    words = []
    previous_end = segment["start"]

    for i, w in enumerate(raw):
        start = w.get("start")
        end = w.get("end")
        if start is None:
            start = previous_end
        if end is None:
            # Untimed words from here up to the next timed start share the gap.
            run = 1
            next_start = segment["end"]
            for n in raw[i + 1:]:
                if n.get("start") is not None:
                    next_start = n["start"]
                    break
                run += 1
            end = start + max(0.0, next_start - start) / run

        words.append({"word": w["word"].strip(), "start": start, "end": end})
        previous_end = end

    return words


def diarize(audio_path, output_path=None, num_speakers=2):
    """Transcribe and diarize audio_path, write the JSON result and return it."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Diarization uses the gated pyannote models: accept "
            "their terms on huggingface.co, then run `export HF_TOKEN=hf_...`."
        )

    audio_path = Path(audio_path)
    if output_path is None:
        output_path = OUTPUT_DIR / f"{audio_path.stem}_transcription.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    print(f"Device: {DEVICE} ({COMPUTE_TYPE})")

    print("[1/5] Loading audio...")
    t = time.perf_counter()
    audio = whisperx.load_audio(str(audio_path))
    _stage(f"{len(audio) / 16000:.0f}s of audio", t)

    print(f"[2/5] Transcribing (whisper-{WHISPER_MODEL})...")
    t = time.perf_counter()
    model = get_transcription_model()
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
    _stage(f"{len(result['segments'])} segments", t)

    print("[3/5] Aligning word timestamps...")
    t = time.perf_counter()
    align_model, metadata = get_align_model()
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, DEVICE,
        return_char_alignments=False,
    )
    _stage("alignment", t)

    print(f"[4/5] Diarizing ({num_speakers} speakers)...")
    t = time.perf_counter()
    diarize_model = get_diarization_model(hf_token)
    diarize_segments = diarize_model(audio, num_speakers=num_speakers)
    _stage("diarization", t)

    print("[5/5] Assigning speakers and writing output...")
    t = time.perf_counter()
    result = whisperx.assign_word_speakers(diarize_segments, result)

    output = {
        "file": audio_path.name,
        "segments": [
            {
                "speaker": segment.get("speaker", "UNKNOWN"),
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "words": _clean_words(segment),
            }
            for segment in result["segments"]
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    _stage(f"wrote {output_path}", t)

    print(f"Total time: {time.perf_counter() - total_start:.1f}s")
    return output


def main():
    parser = argparse.ArgumentParser(description="Transcribe and diarize audio files.")
    parser.add_argument(
        "audio_path", nargs="?",
        help="Path to an audio file (default: every audio file in data/audio/)",
    )
    parser.add_argument(
        "--out", help="Output JSON path (default: output/transcription/<audio_stem>_transcription.json)"
    )
    parser.add_argument(
        "--num-speakers", type=int, default=2, help="Number of speakers (default: 2)"
    )
    args = parser.parse_args()

    if args.audio_path:
        if not os.path.isfile(args.audio_path):
            sys.exit(f"Audio file not found: {args.audio_path}")
        audio_files = [Path(args.audio_path)]
    else:
        audio_files = find_audio_files()
        if not audio_files:
            sys.exit(f"No audio files found in {AUDIO_DIR}")
        if args.out:
            sys.exit("--out can only be used with a single audio_path")
        print(f"Found {len(audio_files)} audio file(s) in {AUDIO_DIR}")

    try:
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n=== [{i}/{len(audio_files)}] {audio_file.name} ===")
            diarize(audio_file, args.out, args.num_speakers)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
