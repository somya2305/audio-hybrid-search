"""Transcribe, align and diarize audio with WhisperX.

Usage:
    python -m src.diarization.diarize [audio_path] [--out path] [--num-speakers N]

With no audio_path, every audio file in data/audio/ is processed.
"""

import argparse
import json
import os
import sys
import time
import warnings
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

# pyannote warns at import that torchcodec can't load FFmpeg; audio is passed
# in memory from whisperx.load_audio, so torchcodec is never used.
warnings.filterwarnings("ignore", message=r"\s*torchcodec is not installed correctly")

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline

from src.common import AUDIO_DIR, TRANSCRIPTION_DIR

AUDIO_EXTENSIONS = {".wav", ".opus", ".mp3", ".m4a", ".flac", ".ogg"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_MODEL = "small"
LANGUAGE = "en"
BATCH_SIZE = 16


@contextmanager
def _stage(label, timings):
    print(f"  {label}...", flush=True)
    start = time.perf_counter()
    yield
    timings[label.split()[0]] = time.perf_counter() - start


# Cached so processing several files loads each model only once.

@lru_cache(maxsize=1)
def get_transcription_model():
    return whisperx.load_model(WHISPER_MODEL, DEVICE, compute_type=COMPUTE_TYPE, language=LANGUAGE)


@lru_cache(maxsize=1)
def get_align_model():
    return whisperx.load_align_model(language_code=LANGUAGE, device=DEVICE)


@lru_cache(maxsize=1)
def get_diarization_model(hf_token):
    return DiarizationPipeline(token=hf_token, device=DEVICE)


def find_audio_files(audio_dir=AUDIO_DIR):
    """Audio files in audio_dir, sorted by name."""
    return sorted(
        p for p in Path(audio_dir).iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _clean_words(segment):
    """The segment's words, each with a start, end and speaker.

    WhisperX leaves some tokens (e.g. numbers) without timestamps. A missing
    start begins where the previous word ended; a missing end runs to the next
    timed word (or the segment end), shared evenly by consecutive untimed
    words. Words without a speaker inherit the previous word's.
    """
    raw = [w for w in segment.get("words", []) if w.get("word", "").strip()]
    words = []
    previous_end = segment["start"]
    previous_speaker = segment.get("speaker", "UNKNOWN")

    for i, w in enumerate(raw):
        start = w["start"] if w.get("start") is not None else previous_end
        end = w.get("end")
        if end is None:
            run, next_start = 1, segment["end"]
            for n in raw[i + 1:]:
                if n.get("start") is not None:
                    next_start = n["start"]
                    break
                run += 1
            end = start + max(0.0, next_start - start) / run

        speaker = w.get("speaker", previous_speaker)
        words.append({"word": w["word"].strip(), "start": start, "end": end, "speaker": speaker})
        previous_end, previous_speaker = end, speaker

    return words


def split_segment_by_speaker(segment):
    """Split a WhisperX segment wherever the word-level speaker changes.

    A segment's speaker is a majority vote over its words, so it can contain
    both speakers. Single-word speaker flips are treated as diarization noise
    and kept with the surrounding run.
    """
    words = _clean_words(segment)
    if not words:
        return [{
            "speaker": segment.get("speaker", "UNKNOWN"),
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"].strip(),
            "words": [],
        }]

    runs = []
    for word in words:
        if runs and runs[-1]["speaker"] == word["speaker"]:
            runs[-1]["words"].append(word)
        else:
            runs.append({"speaker": word["speaker"], "words": [word]})

    smoothed = []
    for run in runs:
        if smoothed and (len(run["words"]) == 1 or smoothed[-1]["speaker"] == run["speaker"]):
            smoothed[-1]["words"].extend(run["words"])
        else:
            smoothed.append(run)

    # A leading single-word run joins the run after it.
    if len(smoothed) > 1 and len(smoothed[0]["words"]) == 1:
        first = smoothed.pop(0)
        smoothed[0]["words"] = first["words"] + smoothed[0]["words"]

    return [
        {
            "speaker": run["speaker"],
            "start": run["words"][0]["start"],
            "end": run["words"][-1]["end"],
            "text": " ".join(w["word"] for w in run["words"]),
            "words": [{"word": w["word"], "start": w["start"], "end": w["end"]} for w in run["words"]],
        }
        for run in smoothed
    ]


def diarize(audio_path, output_path=None, num_speakers=2):
    """Transcribe and diarize audio_path, write the transcript JSON and return it."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Diarization uses the gated pyannote models: accept "
            "their terms on huggingface.co and set HF_TOKEN in .env."
        )

    audio_path = Path(audio_path)
    output_path = Path(output_path or TRANSCRIPTION_DIR / f"{audio_path.stem}_transcription.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timings = {}

    with _stage("transcribing", timings):
        audio = whisperx.load_audio(str(audio_path))
        result = get_transcription_model().transcribe(audio, batch_size=BATCH_SIZE, language=LANGUAGE)
    with _stage("aligning words", timings):
        align_model, metadata = get_align_model()
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, DEVICE, return_char_alignments=False
        )
    with _stage(f"diarizing ({num_speakers} speakers)", timings):
        diarize_segments = get_diarization_model(hf_token)(audio, num_speakers=num_speakers)
    with _stage("assigning speakers", timings):
        result = whisperx.assign_word_speakers(diarize_segments, result)
        output = {
            "file": audio_path.name,
            "segments": [
                split
                for segment in result["segments"]
                for split in split_segment_by_speaker(segment)
            ],
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    stage_times = ", ".join(f"{name} {secs:.1f}s" for name, secs in timings.items())
    print(f"  wrote {output_path.name} ({stage_times}; {sum(timings.values()):.1f}s total)")
    return output


def main():
    parser = argparse.ArgumentParser(description="Transcribe and diarize audio files.")
    parser.add_argument(
        "audio_path", nargs="?", help="Audio file (default: every audio file in data/audio/)"
    )
    parser.add_argument(
        "--out", help="Output JSON path (default: output/transcription/<stem>_transcription.json)"
    )
    parser.add_argument("--num-speakers", type=int, default=2, help="Number of speakers (default: 2)")
    args = parser.parse_args()

    if args.audio_path:
        if not os.path.isfile(args.audio_path):
            sys.exit(f"Audio file not found: {args.audio_path}")
        audio_files = [Path(args.audio_path)]
    else:
        if args.out:
            sys.exit("--out can only be used with a single audio_path")
        audio_files = find_audio_files()
        if not audio_files:
            sys.exit(f"No audio files found in {AUDIO_DIR}")

    try:
        for audio_file in audio_files:
            print(audio_file.name)
            diarize(audio_file, args.out, args.num_speakers)
    except RuntimeError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
