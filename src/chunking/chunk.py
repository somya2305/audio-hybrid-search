"""Group a diarized transcript into speaker-turn chunks.

Usage:
    python -m src.chunking.chunk [transcription.json]

With no path, every *_transcription.json in output/transcription/ is chunked.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
TRANSCRIPTION_DIR = ROOT_DIR / "output" / "transcription"
OUTPUT_DIR = ROOT_DIR / "output" / "chunks"

# A chunk is one speaker turn: consecutive segments by the same speaker.
# Long turns are split at segment (sentence) boundaries once they pass this.
MAX_TURN_WORDS = 80

# Each chunk is embedded together with this many neighbouring turns on each
# side, so a reply ("Yes, that's why we moved") carries what it's replying to.
CONTEXT_TURNS = 1

# Turns shorter than this ("Oh, really?") embed only their own text. With
# neighbour context they would borrow the neighbours' meaning and crowd
# semantic results; they stay keyword-searchable either way.
SHORT_TURN_WORDS = 8


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


def turn_text(turn):
    return " ".join(segment["text"].strip() for segment in turn["segments"])


def create_chunk(turns, index, conversation_id):
    turn = turns[index]
    segments = turn["segments"]
    text = turn_text(turn)

    if turn["word_count"] < SHORT_TURN_WORDS:
        embed_text = text
    else:
        context = turns[max(0, index - CONTEXT_TURNS):index + CONTEXT_TURNS + 1]
        embed_text = " ".join(turn_text(t) for t in context)

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
        "text": text,
        "embed_text": embed_text,
        "words": [word for segment in segments for word in segment.get("words", [])],
    }


def create_chunks(transcript):
    """Turn a diarized transcript dict into a list of speaker-turn chunks."""
    turns = group_turns(transcript.get("segments", []))
    return [
        create_chunk(turns, index, transcript["file"])
        for index in range(len(turns))
    ]


def save_chunks(transcript, chunks, input_path):
    """Write chunks to output/chunks/<stem>_chunks.json and return that path."""
    stem = Path(input_path).stem.removesuffix("_transcription")
    output_path = OUTPUT_DIR / f"{stem}_chunks.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"conversation_id": transcript["file"], "num_chunks": len(chunks), "chunks": chunks},
            f, indent=2, ensure_ascii=False,
        )
    return output_path


def chunk_file(input_path, verbose=True):
    """Chunk one transcription JSON, write <stem>_chunks.json and return the chunks."""
    input_path = Path(input_path)
    with open(input_path, encoding="utf-8") as f:
        transcript = json.load(f)

    chunks = create_chunks(transcript)
    output_path = save_chunks(transcript, chunks, input_path)

    if verbose:
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
    else:
        print(
            f"{transcript['file']:<22} {len(transcript.get('segments', [])):>4} segments -> "
            f"{len(chunks):>4} chunks  ({output_path.relative_to(ROOT_DIR)})"
        )

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Chunk diarized transcripts into speaker turns.")
    parser.add_argument(
        "transcription", nargs="?",
        help="Path to a <stem>_transcription.json file "
             "(default: every *_transcription.json in output/transcription/)",
    )
    args = parser.parse_args()

    if args.transcription:
        input_path = Path(args.transcription)
        if not input_path.is_file():
            sys.exit(f"Transcription file not found: {input_path}")
        chunk_file(input_path)
        return

    input_paths = sorted(TRANSCRIPTION_DIR.glob("*_transcription.json"))
    if not input_paths:
        sys.exit(f"No *_transcription.json files found in {TRANSCRIPTION_DIR}")

    print(f"Chunking {len(input_paths)} transcript(s) from {TRANSCRIPTION_DIR.relative_to(ROOT_DIR)}/")
    for input_path in input_paths:
        chunk_file(input_path, verbose=False)


if __name__ == "__main__":
    main()
