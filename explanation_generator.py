import json
import time
import os

from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL
from style_profiler import StyleProfile

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set. Add it to your .env file.")
    _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def load_eval_profiles(path: str = "eval_style_profiles.json") -> dict[str, StyleProfile]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: StyleProfile(**fields) for name, fields in raw.items()}


STYLE_PROMPT_TEMPLATE = """\
You are a skilled writer who can adapt to any writing style. Explain a concept from a document, perfectly matching the target style.

CRITICAL STYLE CONSTRAINTS (follow these precisely):
- Sentence length: Average {avg_sentence_length} words per sentence. {sentence_length_instruction}
- Grade level: Flesch-Kincaid grade {flesch_kincaid_grade}. {grade_instruction}
- Formality: {formality_score} (scale: <1.0 = very casual/conversational, 1.0-2.0 = moderate, >2.0 = formal/professional, >3.0 = highly formal/academic). {formality_instruction}
- Analogies: {analogy_instruction}
- Questions: {question_instruction}
- Paragraph length: ~{avg_paragraph_length} sentences per paragraph.
- Jargon: {jargon_instruction}

IMPORTANT: Do NOT repeat the same words or phrases. Vary your vocabulary. Each sentence must add new information.

CONTEXT (from the source document):
{context}

CONCEPT TO EXPLAIN: {concept}

Write 2-4 paragraphs explaining the concept. Output ONLY the explanation."""

BASELINE_PROMPT_TEMPLATE = """\
Explain the following concept clearly and accurately for a general audience.

CONTEXT (from the source document):
{context}

CONCEPT TO EXPLAIN: {concept}

Write 2-3 paragraphs. Output ONLY the explanation."""


class ExplanationGenerator:
    def __init__(self, model_name: str = GEMINI_MODEL, temperature: float = 0.7):
        self.client = _get_client()
        self.model_name = model_name
        self.temperature = temperature

    def _call_api(self, prompt: str, retries: int = 3, delay: float = 5.0) -> str:
        """Call Gemini API with retry logic for rate limits and server errors."""
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={"temperature": self.temperature},
                )
                return response.text.strip()
            except Exception as e:
                if attempt < retries - 1:
                    wait = delay * (attempt + 1)  # increasing backoff
                    print(f"  (retry {attempt+1}/{retries-1}: {e})")
                    print(f"  (waiting {wait}s...)")
                    time.sleep(wait)
                else:
                    print(f"  (failed after {retries} attempts: {e})")
                    return "[generation failed]"
        return "[generation failed]"

    def generate(self, concept: str, context: str, style_profile: StyleProfile) -> str:
        """Generate a style-adapted explanation."""
        if style_profile.avg_sentence_length < 12:
            sentence_length_instruction = "Keep sentences SHORT and punchy."
        elif style_profile.avg_sentence_length > 30:
            sentence_length_instruction = "Use LONG, complex, multi-clause sentences."
        else:
            sentence_length_instruction = "Use medium-length sentences."

        if style_profile.flesch_kincaid_grade < 8:
            grade_instruction = "Use simple everyday words a middle schooler would know."
        elif style_profile.flesch_kincaid_grade > 15:
            grade_instruction = "Use sophisticated vocabulary and complex sentence structures."
        else:
            grade_instruction = "Use moderately complex language."

        if style_profile.formality_score < 1.5:
            formality_instruction = "Write casually. Use contractions, 'you' and 'we', colloquial phrasing."
        elif style_profile.formality_score > 2.5:
            formality_instruction = "Write in a highly formal, impersonal, academic register. Use passive voice. Avoid 'you' and 'I'."
        else:
            formality_instruction = "Write in a balanced professional tone."

        if style_profile.uses_analogies:
            analogy_instruction = "MUST include at least 1-2 analogies or comparisons using phrases like 'like', 'similar to', 'think of it as', 'imagine'."
        else:
            analogy_instruction = "Do NOT use any analogies, metaphors, or comparisons. Be direct and literal."

        if style_profile.uses_questions:
            question_instruction = "MUST include 1-2 rhetorical questions to engage the reader."
        else:
            question_instruction = "Do NOT use any questions. Use only declarative statements."

        if style_profile.jargon_comfort > 0.2:
            jargon_instruction = "Use technical terminology freely without defining it — assume the reader is an expert."
        elif style_profile.jargon_comfort < 0.1:
            jargon_instruction = "AVOID technical jargon. If you must use a technical term, immediately explain it in plain language."
        else:
            jargon_instruction = "Use some technical terms but briefly clarify complex ones."

        prompt = STYLE_PROMPT_TEMPLATE.format(
            avg_sentence_length=style_profile.avg_sentence_length,
            sentence_length_instruction=sentence_length_instruction,
            flesch_kincaid_grade=style_profile.flesch_kincaid_grade,
            grade_instruction=grade_instruction,
            formality_score=style_profile.formality_score,
            formality_instruction=formality_instruction,
            analogy_instruction=analogy_instruction,
            question_instruction=question_instruction,
            avg_paragraph_length=style_profile.avg_paragraph_length,
            jargon_instruction=jargon_instruction,
            context=context,
            concept=concept,
        )

        return self._call_api(prompt)

    def generate_baseline(self, concept: str, context: str) -> str:
        """Generate a baseline explanation with no style adaptation."""
        prompt = BASELINE_PROMPT_TEMPLATE.format(context=context, concept=concept)
        return self._call_api(prompt)


if __name__ == "__main__":
    profiles = load_eval_profiles()

    test_concept = "backpropagation"
    test_context = (
        "Neural networks learn by adjusting the weights of connections between neurons. "
        "During training, the network makes a prediction, compares it to the true label "
        "using a loss function, and then propagates the error backward through the layers "
        "to update each weight via gradient descent. This backward pass computes partial "
        "derivatives of the loss with respect to each weight using the chain rule of calculus, "
        "allowing even deep networks with many layers to learn effectively."
    )

    generator = ExplanationGenerator()

    for profile_name, profile in profiles.items():
        print(f"\n{'=' * 60}")
        print(f"PROFILE: {profile_name}")
        print(f"{'=' * 60}")
        explanation = generator.generate(test_concept, test_context, profile)
        print(explanation)
        time.sleep(3)  # rate limit buffer

    print(f"\n{'=' * 60}")
    print("BASELINE (no style adaptation)")
    print(f"{'=' * 60}")
    baseline = generator.generate_baseline(test_concept, test_context)
    print(baseline)