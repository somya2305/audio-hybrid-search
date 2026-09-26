"""Group a diarized transcript into speaker-turn chunks.

Usage:
    python -m src.chunking.chunk <transcription.json>
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT_DIR / "output" / "chunks"

# A chunk is one speaker turn: consecutive segments by the same speaker.
# Long turns are split at segment (sentence) boundaries once they pass this.
MAX_TURN_WORDS = 80


def word_count(text):
    return len(re.findall(r"\b\w+\b", text))


def group_turns(segments):
    """Group consecutive same-speaker segments into turns of <= MAX_TURN_WORDS.

    A single segment longer than MAX_TURN_WORDS stays whole, since turns are
    only split at segment boundaries.
    """
    turns = []

    for segment in segments:
        text = segment.get("text", "").strip()
        if not text:
            continue

        speaker = segment.get("speaker", "UNKNOWN")
        words = word_count(text)

        if (
            turns
            and turns[-1]["speaker"] == speaker
            and turns[-1]["word_count"] + words <= MAX_TURN_WORDS
        ):
            turns[-1]["segments"].append(segment)
            turns[-1]["word_count"] += words
        else:
            turns.append({"speaker": speaker, "segments": [segment], "word_count": words})

    return turns


def create_chunk(turn, index, conversation_id):
    segments = turn["segments"]
    start = segments[0]["start"]
    end = segments[-1]["end"]

    return {
        "chunk_id": f"{conversation_id}_{index + 1:04d}",
        "conversation_id": conversation_id,
        "chunk_index": index + 1,
        "speaker": turn["speaker"],
        "start": round(start, 2),
        "end": round(end, 2),
        "duration": round(end - start, 2),
        "word_count": turn["word_count"],
        "text": " ".join(segment["text"].strip() for segment in segments),
        "words": [word for segment in segments for word in segment.get("words", [])],
    }


def create_chunks(transcript):
    """Turn a diarized transcript dict into a list of speaker-turn chunks."""
    turns = group_turns(transcript.get("segments", []))
    return [
        create_chunk(turn, index, transcript["file"])
        for index, turn in enumerate(turns)
    ]


def main():
    parser = argparse.ArgumentParser(description="Chunk a diarized transcript into speaker turns.")
    parser.add_argument("transcription", help="Path to a <stem>_transcription.json file")
    args = parser.parse_args()

    input_path = Path(args.transcription)
    if not input_path.is_file():
        sys.exit(f"Transcription file not found: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        transcript = json.load(f)

    chunks = create_chunks(transcript)

    stem = input_path.stem.removesuffix("_transcription")
    output_path = OUTPUT_DIR / f"{stem}_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"conversation_id": transcript["file"], "num_chunks": len(chunks), "chunks": chunks},
            f, indent=2, ensure_ascii=False,
        )

    print(f"Conversation : {transcript['file']}")
    print(f"Segments     : {len(transcript.get('segments', []))}")
    print(f"Chunks       : {len(chunks)}")
    for chunk in chunks:
        print(
            f"{chunk['chunk_id']} | "
            f"{chunk['start']}s - {chunk['end']}s | "
            f"{chunk['word_count']} words | "
            f"speaker={chunk['speaker']}"
        )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
