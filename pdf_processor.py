"""
PDF Processor for Refract.
Parses PDFs with pdfplumber, chunks by detected headings (falls back to
paragraph-level splits), and stores chunks + metadata in ChromaDB.
"""

import pdfplumber
import chromadb
import re
import os
from dataclasses import dataclass
from typing import Optional
from collections import Counter
from config import CHROMA_PATH


@dataclass
class TextLine:
    """A single line extracted from a PDF with font metadata."""
    text: str
    font_size: float
    is_bold: bool
    page_num: int


@dataclass
class Chunk:
    """A semantically meaningful chunk of PDF text."""
    text: str
    page_num: int
    section_title: Optional[str]
    chunk_index: int


class FallbackEmbedding:
    """Simple hash-based embedding for dev. ChromaDB's default (ONNX MiniLM)
    is better — it will work on your machine where the model can download.
    To use the default, just remove the embedding_function arg from create_collection.
    """

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            words = text.lower().split()
            vec = [0.0] * 384
            for word in words:
                vec[hash(word) % 384] += 1.0
            norm = sum(v * v for v in vec) ** 0.5
            embeddings.append([v / norm if norm else 0.0 for v in vec])
        return embeddings


class PDFProcessor:
    def __init__(self, chroma_path: str = CHROMA_PATH, use_fallback_embed: bool = False):
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.embed_fn = FallbackEmbedding() if use_fallback_embed else None

    def process(self, pdf_path: str) -> list[Chunk]:
        """Main entry: parse PDF -> chunk -> store in ChromaDB -> return chunks."""
        lines = self._extract_lines(pdf_path)

        if not lines:
            raise ValueError(f"No text extracted from {pdf_path}")

        body_size = self._get_body_font_size(lines)
        heading_indices = self._detect_headings(lines, body_size)

        if len(heading_indices) >= 2:
            chunks = self._chunk_by_headings(lines, heading_indices)
            method = "headings"
        else:
            full_text = "\n".join(l.text for l in lines)
            chunks = self._chunk_by_paragraphs(full_text, lines)
            method = "paragraphs"

        # Drop tiny chunks
        chunks = [c for c in chunks if len(c.text.strip()) > 50]
        for i, chunk in enumerate(chunks):
            chunk.chunk_index = i

        collection_name = self._sanitize_collection_name(pdf_path)
        self._store_in_chroma(chunks, collection_name)

        print(f"[{method}] {len(chunks)} chunks from {os.path.basename(pdf_path)}")
        return chunks

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------

    def _extract_lines(self, pdf_path: str) -> list[TextLine]:
        """Extract every line in the PDF with font size + bold metadata."""
        all_lines: list[TextLine] = []

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                chars = page.chars

                if not chars:
                    # Scanned page or no char data — plain text fallback
                    text = page.extract_text() or ""
                    for line_text in text.split("\n"):
                        if line_text.strip():
                            all_lines.append(TextLine(line_text.strip(), 12.0, False, page_num))
                    continue

                # Group chars into lines by vertical position
                sorted_chars = sorted(chars, key=lambda c: (round(c["top"], 1), c["x0"]))
                line_groups: list[list[dict]] = []
                current: list[dict] = [sorted_chars[0]]

                for char in sorted_chars[1:]:
                    if abs(char["top"] - current[0]["top"]) < 4:
                        current.append(char)
                    else:
                        line_groups.append(current)
                        current = [char]
                line_groups.append(current)

                for group in line_groups:
                    group.sort(key=lambda c: c["x0"])
                    text = "".join(c["text"] for c in group).strip()
                    if not text:
                        continue

                    sizes = [c["size"] for c in group if c["text"].strip()]
                    font_size = Counter(sizes).most_common(1)[0][0] if sizes else 12.0
                    is_bold = any("Bold" in (c.get("fontname") or "") for c in group)

                    all_lines.append(TextLine(text, font_size, is_bold, page_num))

        return all_lines

    # ------------------------------------------------------------------
    # Heading detection
    # ------------------------------------------------------------------

    def _get_body_font_size(self, lines: list[TextLine]) -> float:
        sizes = [round(l.font_size, 1) for l in lines if l.text.strip()]
        return Counter(sizes).most_common(1)[0][0] if sizes else 12.0

    def _detect_headings(self, lines: list[TextLine], body_size: float) -> list[int]:
        """Return indices of lines that look like section headings."""
        indices: list[int] = []

        for i, line in enumerate(lines):
            text = line.text.strip()
            if not text or len(text) > 200:
                continue

            is_heading = False

            # 1. Font size noticeably larger than body
            if line.font_size > body_size * 1.15:
                is_heading = True
            # 2. ALL CAPS multi-word short line
            elif text.isupper() and len(text) < 100 and " " in text:
                is_heading = True
            # 3. Bold + short + doesn't end with period
            elif line.is_bold and len(text) < 80 and not text.endswith("."):
                is_heading = True

            if is_heading:
                indices.append(i)

        return indices

    # ------------------------------------------------------------------
    # Chunking strategies
    # ------------------------------------------------------------------

    def _chunk_by_headings(self, lines: list[TextLine], heading_indices: list[int]) -> list[Chunk]:
        chunks: list[Chunk] = []

        # Text before the first heading
        if heading_indices[0] > 0:
            pre = lines[: heading_indices[0]]
            pre_text = "\n".join(l.text for l in pre).strip()
            if len(pre_text) > 50:
                chunks.append(Chunk(pre_text, pre[0].page_num, None, 0))

        for i, h_idx in enumerate(heading_indices):
            end_idx = heading_indices[i + 1] if i + 1 < len(heading_indices) else len(lines)
            section_title = lines[h_idx].text.strip()
            body_lines = lines[h_idx + 1 : end_idx]

            if not body_lines:
                continue

            text = "\n".join(l.text for l in body_lines).strip()
            chunks.append(Chunk(text, lines[h_idx].page_num, section_title, len(chunks)))

        return chunks

    def _chunk_by_paragraphs(self, full_text: str, lines: list[TextLine]) -> list[Chunk]:
        """Split on blank lines, merge small paragraphs, add 1-sentence overlap."""
        paragraphs = re.split(r"\n\s*\n", full_text)
        paragraphs = [p.strip() for p in paragraphs if len(p.strip()) > 30]

        if not paragraphs:
            return [Chunk(full_text.strip(), 1, None, 0)]

        # Merge small paragraphs to hit ~500-1500 chars per chunk
        merged: list[str] = []
        buf = ""
        for para in paragraphs:
            if len(buf) + len(para) < 1500:
                buf += ("\n\n" + para if buf else para)
            else:
                if buf:
                    merged.append(buf)
                buf = para
        if buf:
            merged.append(buf)

        # Build chunks
        chunks: list[Chunk] = []
        for i, text in enumerate(merged):
            page = self._estimate_page(text, lines)
            chunks.append(Chunk(text, page, None, i))

        # 1-sentence overlap between consecutive chunks
        for i in range(1, len(chunks)):
            prev_sentences = re.split(r"(?<=[.!?])\s+", chunks[i - 1].text)
            if prev_sentences:
                chunks[i].text = prev_sentences[-1] + " " + chunks[i].text

        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _estimate_page(self, text_fragment: str, lines: list[TextLine]) -> int:
        start = text_fragment[:60]
        for line in lines:
            if start.startswith(line.text[:40]):
                return line.page_num
        return 1

    def _sanitize_collection_name(self, pdf_path: str) -> str:
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)[:50]
        if not name or not name[0].isalnum():
            name = "doc_" + name
        if not name[-1].isalnum():
            name = name.rstrip("_-") or name + "0"
        # ChromaDB requires length >= 3
        while len(name) < 3:
            name += "0"
        return name

    def _store_in_chroma(self, chunks: list[Chunk], collection_name: str):
        try:
            self.chroma_client.delete_collection(name=collection_name)
        except Exception:
            pass

        kwargs = {"name": collection_name}
        if self.embed_fn:
            kwargs["embedding_function"] = self.embed_fn
        collection = self.chroma_client.create_collection(**kwargs)

        collection.add(
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "page_num": c.page_num,
                    "section_title": c.section_title or "untitled",
                    "chunk_index": c.chunk_index,
                }
                for c in chunks
            ],
            ids=[f"chunk_{c.chunk_index}" for c in chunks],
        )


# ------------------------------------------------------------------
# CLI test
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pdf_processor.py <pdf_path>")
        sys.exit(1)

    processor = PDFProcessor(use_fallback_embed=True)
    chunks = processor.process(sys.argv[1])

    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}\n")

    for chunk in chunks:
        title = chunk.section_title or "N/A"
        preview = chunk.text[:200].replace("\n", " ")
        print(f"[Chunk {chunk.chunk_index}] Page {chunk.page_num} | {title}")
        print(f"  {preview}...")
        print()