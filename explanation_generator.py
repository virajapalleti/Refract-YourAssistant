import json
import os

from google import genai

from config import GEMINI_API_KEY, GEMINI_MODEL
from style_profiler import StyleProfile
from concept_extractor import ConceptExtractor as ConceptResult

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
You are explaining a concept from a document. Match the following writing style EXACTLY:

TARGET STYLE:
- Average sentence length: {avg_sentence_length} words (stay within ±2 words)
- Flesch-Kincaid grade level: {flesch_kincaid_grade} (stay within ±1 grade)
- Formality score: {formality_score} (higher = more formal noun/preposition-heavy language, lower = pronoun/verb-heavy conversational)
- Uses analogies: {uses_analogies} ({analogy_instruction})
- Uses questions: {uses_questions} ({question_instruction})
- Average paragraph length: {avg_paragraph_length} sentences per paragraph
- Jargon comfort: {jargon_comfort} (higher = use technical terminology freely, lower = avoid words with 3+ syllables when possible)
- Vocabulary richness: {vocabulary_richness}

CONTEXT (from the source document):
{context}

CONCEPT TO EXPLAIN: {concept}

Write an explanation of the concept in the target style. Output ONLY the explanation, no preamble, no meta-commentary."""

BASELINE_PROMPT_TEMPLATE = """\
CONTEXT (from the source document):
{context}

CONCEPT TO EXPLAIN: {concept}

Explain this concept clearly and accurately. Output ONLY the explanation, no preamble, no meta-commentary."""


class ExplanationGenerator:
    def __init__(self, model_name: str = GEMINI_MODEL, temperature: float = 0.7):
        self.client = _get_client()
        self.model_name = model_name
        self.temperature = temperature

    def generate(self, concept: str, context: str, style_profile: StyleProfile) -> str:
        analogy_instruction = (
            "Include at least one analogy using 'like', 'imagine', or 'think of it as'"
            if style_profile.uses_analogies
            else "Do NOT use analogies or comparisons"
        )
        question_instruction = (
            "Include at least one rhetorical question"
            if style_profile.uses_questions
            else "Do NOT use rhetorical questions"
        )

        prompt = STYLE_PROMPT_TEMPLATE.format(
            avg_sentence_length=style_profile.avg_sentence_length,
            flesch_kincaid_grade=style_profile.flesch_kincaid_grade,
            formality_score=style_profile.formality_score,
            uses_analogies=style_profile.uses_analogies,
            analogy_instruction=analogy_instruction,
            uses_questions=style_profile.uses_questions,
            question_instruction=question_instruction,
            avg_paragraph_length=style_profile.avg_paragraph_length,
            jargon_comfort=style_profile.jargon_comfort,
            vocabulary_richness=style_profile.vocabulary_richness,
            context=context,
            concept=concept,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"temperature": self.temperature},
        )
        return response.text

    def generate_baseline(self, concept: str, context: str) -> str:
        prompt = BASELINE_PROMPT_TEMPLATE.format(context=context, concept=concept)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"temperature": self.temperature},
        )
        return response.text


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

    print(f"\n{'=' * 60}")
    print("BASELINE (no style adaptation)")
    print(f"{'=' * 60}")
    baseline = generator.generate_baseline(test_concept, test_context)
    print(baseline)
