"""
PDF Processor for Refract.
Parses PDFs with pdfplumber, chunks by detected headings (falls back to
paragraph-level splits), and stores chunks + metadata in ChromaDB.
"""

import os
import re
import sys
import statistics
from dataclasses import dataclass, replace
from typing import Optional

import pdfplumber
import chromadb

from config import CHROMA_PATH

LINE_TOLERANCE = 4  # px — chars within this vertical distance are the same line
TARGET_CHUNK_SIZE = 1500  # chars, for paragraph fallback
MIN_CHUNK_CHARS = 50
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A semantically meaningful chunk of PDF text."""
    text: str
    page_num: int
    section_title: Optional[str]
    chunk_index: int


@dataclass
class _Line:
    """A single line extracted from a PDF with font metadata."""
    text: str
    size: float
    is_bold: bool
    page_num: int
    top: float


class PDFProcessor:
    def __init__(self, chroma_path: str = CHROMA_PATH):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)

    def process(self, pdf_path: str) -> list:
        """Main entry: parse PDF -> chunk -> store in ChromaDB -> return chunks."""
        with pdfplumber.open(pdf_path) as pdf:
            lines = self._extract_lines(pdf)
            if not lines:
                raise ValueError(f"No text extracted from {pdf_path}")

            body_size = self._body_font_size(lines)
            heading_lines = [l for l in lines if self._is_heading(l, body_size)]

            if len(heading_lines) >= 2:
                chunks = self._chunk_by_headings(lines, body_size)
                method = "headings"
            else:
                chunks = self._chunk_by_paragraphs(pdf)
                method = "paragraphs"

        chunks = [c for c in chunks if len(c.text.strip()) >= MIN_CHUNK_CHARS]
        chunks = [replace(c, chunk_index=i) for i, c in enumerate(chunks)]

        self._store(chunks, pdf_path)
        print(f"[{method}] {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        return chunks

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_lines(self, pdf) -> list:
        """Group each page's chars into lines by vertical position."""
        lines = []
        for page_num, page in enumerate(pdf.pages, start=1):
            chars = page.chars
            if not chars:
                continue

            chars_sorted = sorted(chars, key=lambda c: c["top"])
            groups = []
            current_group = [chars_sorted[0]]
            current_top = chars_sorted[0]["top"]
            for c in chars_sorted[1:]:
                if abs(c["top"] - current_top) <= LINE_TOLERANCE:
                    current_group.append(c)
                else:
                    groups.append(current_group)
                    current_group = [c]
                    current_top = c["top"]
            groups.append(current_group)

            for group in groups:
                group_sorted = sorted(group, key=lambda c: c["x0"])
                text = "".join(c["text"] for c in group_sorted).strip()
                if not text:
                    continue

                sizes = [round(c["size"], 1) for c in group_sorted]
                try:
                    size_mode = statistics.mode(sizes)
                except statistics.StatisticsError:
                    size_mode = sizes[0]

                fontnames = [c["fontname"] for c in group_sorted]
                try:
                    fontname_mode = statistics.mode(fontnames)
                except statistics.StatisticsError:
                    fontname_mode = fontnames[0]
                is_bold = "bold" in fontname_mode.lower()

                lines.append(
                    _Line(
                        text=text,
                        size=size_mode,
                        is_bold=is_bold,
                        page_num=page_num,
                        top=group_sorted[0]["top"],
                    )
                )
        return lines

    # ------------------------------------------------------------------
    # Heading detection
    # ------------------------------------------------------------------

    def _body_font_size(self, lines: list) -> float:
        sizes = [l.size for l in lines]
        if not sizes:
            return 0.0
        try:
            return statistics.mode(sizes)
        except statistics.StatisticsError:
            return sizes[0]

    def _is_heading(self, line: "_Line", body_size: float) -> bool:
        text = line.text.strip()
        if not text:
            return False

        larger_font = body_size > 0 and line.size > body_size * 1.15
        all_caps_multiword = text.isupper() and len(text.split()) > 1 and len(text) < 100
        bold_short_no_period = line.is_bold and len(text) < 80 and not text.endswith(".")
        return larger_font or all_caps_multiword or bold_short_no_period

    # ------------------------------------------------------------------
    # Chunking strategies
    # ------------------------------------------------------------------

    def _chunk_by_headings(self, lines: list, body_size: float) -> list:
        heading_indices = [i for i, l in enumerate(lines) if self._is_heading(l, body_size)]
        chunks = []
        chunk_index = 0

        if heading_indices[0] > 0:
            intro_lines = lines[: heading_indices[0]]
            text = "\n".join(l.text for l in intro_lines).strip()
            if len(text) >= MIN_CHUNK_CHARS:
                chunks.append(
                    Chunk(text=text, page_num=intro_lines[0].page_num, section_title=None, chunk_index=chunk_index)
                )
                chunk_index += 1

        for idx, h_idx in enumerate(heading_indices):
            heading_line = lines[h_idx]
            end_idx = heading_indices[idx + 1] if idx + 1 < len(heading_indices) else len(lines)
            body_lines = lines[h_idx + 1 : end_idx]
            text = "\n".join(l.text for l in body_lines).strip()
            if len(text) < MIN_CHUNK_CHARS:
                continue
            chunks.append(
                Chunk(
                    text=text,
                    page_num=heading_line.page_num,
                    section_title=heading_line.text.strip(),
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1

        return chunks

    def _chunk_by_paragraphs(self, pdf) -> list:
        """Split on blank lines, merge small paragraphs to ~1500 chars, add 1-sentence overlap."""
        page_texts = [(i, page.extract_text() or "") for i, page in enumerate(pdf.pages, start=1)]

        paragraphs = []
        for page_num, text in page_texts:
            for para in re.split(r"\n\s*\n", text):
                para = para.strip()
                if para:
                    paragraphs.append((page_num, para))

        groups = []
        current_texts = []
        current_page = None
        current_len = 0
        for page_num, para in paragraphs:
            if current_page is None:
                current_page = page_num
            if current_texts and current_len + len(para) > TARGET_CHUNK_SIZE:
                groups.append({"text": "\n\n".join(current_texts), "page_num": current_page})
                current_texts = []
                current_len = 0
                current_page = page_num
            current_texts.append(para)
            current_len += len(para)
        if current_texts:
            groups.append({"text": "\n\n".join(current_texts), "page_num": current_page})

        chunks = []
        for idx, g in enumerate(groups):
            text = g["text"]
            if idx > 0:
                prev_text = groups[idx - 1]["text"].strip()
                sentences = SENTENCE_SPLIT_RE.split(prev_text)
                overlap = sentences[-1].strip() if sentences else ""
                if overlap:
                    text = f"{overlap} {text}"
            chunks.append(Chunk(text=text, page_num=g["page_num"], section_title=None, chunk_index=idx))

        return chunks

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def _sanitize_collection_name(self, pdf_path: str) -> str:
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:50]
        if not name or not name[0].isalnum():
            name = "doc_" + name
        while len(name) < 3:
            name += "0"
        return name

    def _store(self, chunks: list, pdf_path: str):
        collection_name = self._sanitize_collection_name(pdf_path)
        collection = self.chroma_client.get_or_create_collection(name=collection_name)

        if not chunks:
            return collection

        collection.upsert(
            ids=[f"chunk_{c.chunk_index}" for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "page_num": c.page_num,
                    "section_title": c.section_title or "",
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
        )
        return collection


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_path>")
        sys.exit(1)

    processor = PDFProcessor()
    chunks = processor.process(sys.argv[1])

    print(f"\n{'=' * 60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'=' * 60}\n")

    for chunk in chunks:
        title = chunk.section_title or "N/A"
        preview = chunk.text[:200].replace("\n", " ")
        print(f"[Chunk {chunk.chunk_index}] Page {chunk.page_num} | {title}")
        print(f"  {preview}...")
        print()
