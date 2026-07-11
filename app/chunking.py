"""Document chunking (Phase 3: chunking & context management).

Splits long markdown/issue text into overlapping, heading-aware chunks so
retrieval is precise and stays within useful context windows.
"""
from __future__ import annotations

import re

_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)


def _split_sections(text: str) -> list[str]:
    """Split markdown on headings; keep each heading with its body."""
    if not _HEADING_RE.search(text):
        return [text]
    parts, last = [], 0
    for m in _HEADING_RE.finditer(text):
        if m.start() > last:
            parts.append(text[last:m.start()])
        last = m.start()
    parts.append(text[last:])
    return [p for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> list[str]:
    """Heading-aware, paragraph-packed chunks with character overlap."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[str] = []
    for section in _split_sections(text):
        if len(section) <= max_chars:
            chunks.append(section.strip())
            continue
        # pack paragraphs up to max_chars, then carry an overlap tail
        buf = ""
        for para in re.split(r"\n\s*\n", section):
            para = para.strip()
            if not para:
                continue
            if len(buf) + len(para) + 2 <= max_chars:
                buf = f"{buf}\n\n{para}" if buf else para
            else:
                if buf:
                    chunks.append(buf.strip())
                tail = buf[-overlap:] if buf else ""
                buf = f"{tail}\n\n{para}".strip() if tail else para
                # a single oversized paragraph: hard-split it
                while len(buf) > max_chars:
                    chunks.append(buf[:max_chars].strip())
                    buf = buf[max_chars - overlap:]
        if buf.strip():
            chunks.append(buf.strip())
    return [c for c in chunks if c]
