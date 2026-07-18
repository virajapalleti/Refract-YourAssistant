"""
Concept extraction for Refract.
Uses Gemini Flash to identify key concepts a reader would need explained
in order to fully understand each chunk of a document.
"""

import json
import re
import sys
import time

from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL
from pdf_processor import Chunk, PDFProcessor

PROMPT_TEMPLATE = (
    "Given this passage from a document, identify 2-5 key concepts that a reader "
    "would need explained to fully understand the passage. For each concept, provide:\n"
    "1. A short concept name (2-5 words)\n"
    "2. A one-sentence reason why it needs explanation.\n"
    "Return ONLY a JSON array. No markdown, no backticks, no preamble.\n\n"
    "Passage:\n{passage}"
)

MAX_CONCEPTS = 5
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1
RATE_LIMIT_DELAY_SECONDS = 1

FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class ConceptExtractor:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.results = []

    def _parse_response(self, raw_text: str) -> list:
        text = FENCE_RE.sub("", raw_text.strip()).strip()
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("Expected a JSON array")

        validated = [item for item in data if isinstance(item, dict) and "concept" in item]
        return validated[:MAX_CONCEPTS]

    def extract_for_chunk(self, chunk: Chunk) -> list:
        prompt = PROMPT_TEMPLATE.format(passage=chunk.text)
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                )
                return self._parse_response(response.text)
            except (json.JSONDecodeError, ValueError) as e:
                last_error = e
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

        print(f"Warning: failed to parse concepts for chunk {chunk.chunk_index}: {last_error}")
        return []

    def extract(self, chunks: list) -> list:
        self.results = []
        for chunk in chunks:
            concepts = self.extract_for_chunk(chunk)
            self.results.append(
                {
                    "chunk_index": chunk.chunk_index,
                    "page_num": chunk.page_num,
                    "section_title": chunk.section_title,
                    "concepts": concepts,
                    "chunk_preview": chunk.text[:300],
                }
            )
            time.sleep(RATE_LIMIT_DELAY_SECONDS)
        return self.results

    def save_results(self, output_path: str = "concepts.json") -> str:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        return output_path


def print_results(results: list):
    for r in results:
        print(f"\n=== Chunk {r['chunk_index']} (page {r['page_num']}) ===")
        if r["section_title"]:
            print(f"Section: {r['section_title']}")
        preview = r["chunk_preview"][:150].replace("\n", " ")
        print(f"Preview: {preview}...")

        if not r["concepts"]:
            print("  (no concepts extracted)")
        for c in r["concepts"]:
            print(f"  - {c.get('concept')}: {c.get('reason', '')}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python concept_extractor.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    processor = PDFProcessor()
    chunks = processor.process(pdf_path)
    print(f"Extracted {len(chunks)} chunks. Running concept extraction...\n")

    extractor = ConceptExtractor()
    results = extractor.extract(chunks)
    output_path = extractor.save_results()

    print_results(results)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
