import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'baseline'))
from chunking import chunk_text as _chunk_text


def chunk(text: str) -> list[str]:
    return _chunk_text(text, chunk_size=256, overlap=32)
