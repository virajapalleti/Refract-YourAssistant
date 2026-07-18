"""
Style profiler for Refract.
Extracts 8 measurable linguistic features from writing samples using spaCy
and textstat. No LLM calls — purely statistical.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict

import spacy
import textstat

ANALOGY_RE = re.compile(
    r"\blike (?:a|an|the)\b"
    r"|\bas if\b"
    r"|\bas though\b"
    r"|\bimagine\b"
    r"|\bthink of it as\b"
    r"|\bsimilar to\b"
    r"|\bpicture\b"
    r"|\banalog(?:y|ies|ous)\b"
    r"|\bjust like\b"
    r"|\bthe same way\b",
    re.IGNORECASE,
)

FORMAL_POS = {"NOUN", "PROPN", "ADP", "DET"}
CASUAL_POS = {"PRON", "ADV", "INTJ", "VERB"}


@dataclass
class StyleProfile:
    avg_sentence_length: float
    vocabulary_richness: float
    flesch_kincaid_grade: float
    formality_score: float
    uses_analogies: bool
    uses_questions: bool
    avg_paragraph_length: float
    jargon_comfort: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class StyleProfiler:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def profile(self, samples: list) -> StyleProfile:
        combined_text = "\n\n".join(samples)
        doc = self.nlp(combined_text)
        sentences = list(doc.sents)
        num_sentences = len(sentences) or 1

        avg_sentence_length = self._avg_sentence_length(sentences, num_sentences)
        vocabulary_richness = self._vocabulary_richness(doc)
        flesch_kincaid_grade = round(textstat.flesch_kincaid_grade(combined_text), 1)
        formality_score = self._formality_score(doc)
        uses_analogies = self._uses_analogies(combined_text, num_sentences)
        uses_questions = self._uses_questions(sentences, num_sentences)
        avg_paragraph_length = self._avg_paragraph_length(combined_text)
        jargon_comfort = self._jargon_comfort(doc)

        return StyleProfile(
            avg_sentence_length=avg_sentence_length,
            vocabulary_richness=vocabulary_richness,
            flesch_kincaid_grade=flesch_kincaid_grade,
            formality_score=formality_score,
            uses_analogies=uses_analogies,
            uses_questions=uses_questions,
            avg_paragraph_length=avg_paragraph_length,
            jargon_comfort=jargon_comfort,
        )

    def _avg_sentence_length(self, sentences, num_sentences) -> float:
        word_counts = [sum(1 for t in sent if not t.is_punct and not t.is_space) for sent in sentences]
        return round(sum(word_counts) / num_sentences, 1) if word_counts else 0.0

    def _vocabulary_richness(self, doc) -> float:
        alpha_words = [t.text.lower() for t in doc if t.is_alpha]
        if not alpha_words:
            return 0.0
        return round(len(set(alpha_words)) / len(alpha_words), 3)

    def _formality_score(self, doc) -> float:
        pos_counts = Counter(t.pos_ for t in doc)
        formal_count = sum(pos_counts.get(p, 0) for p in FORMAL_POS)
        casual_count = sum(pos_counts.get(p, 0) for p in CASUAL_POS)
        if casual_count == 0:
            return 3.0
        return round(formal_count / casual_count, 2)

    def _uses_analogies(self, text, num_sentences) -> bool:
        matches = len(ANALOGY_RE.findall(text))
        return (matches / num_sentences) >= 0.02

    def _uses_questions(self, sentences, num_sentences) -> bool:
        question_count = sum(1 for s in sentences if s.text.strip().endswith("?"))
        return (question_count / num_sentences) >= 0.05

    def _avg_paragraph_length(self, text) -> float:
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            return 0.0
        sentence_counts = [len(list(self.nlp(p).sents)) for p in paragraphs]
        return round(sum(sentence_counts) / len(paragraphs), 1)

    def _jargon_comfort(self, doc) -> float:
        alpha_words = [t.text for t in doc if t.is_alpha]
        if not alpha_words:
            return 0.0
        long_words = sum(1 for w in alpha_words if textstat.syllable_count(w) >= 3)
        return round(long_words / len(alpha_words), 3)


CASUAL_REDDIT_SAMPLE = """
ngl I've been putting off learning this for way too long lol. finally sat
down today and just started messing around with it, and honestly? not as
bad as I thought. like, everyone online makes it sound so intimidating but
once you actually try stuff instead of just reading docs it clicks pretty
fast. anyway does anyone else feel like tutorials always skip the part
where things actually break? that's like 90% of what you actually learn
from imo. gonna keep grinding on this tonight, wish me luck lol, will
update if I figure out why my thing keeps crashing every five seconds.
"""

ACADEMIC_ABSTRACT_SAMPLE = """
This study examines the extent to which structural interventions influence
long-term behavioral outcomes within distributed systems, employing a
mixed-methods framework in order to reconcile quantitative performance
metrics with qualitative practitioner assessments. Results indicate that
the proposed intervention yields a statistically significant reduction in
observed latency variance, a finding that is consistent across the
majority of evaluated deployment configurations. These outcomes are
discussed in relation to prior work, and implications for future research
methodology are considered, particularly with respect to the
generalizability of the present findings to systems of substantially
greater scale.
"""

CONVERSATIONAL_BLOG_SAMPLE = """
Have you ever wondered why some code just feels easier to work with than
other code? I think about this a lot, and honestly, we don't talk about it
enough. When we write something today, we're really writing a letter to
whoever touches it next, and more often than not that's future you. So
next time you're tempted to skip a clear name or leave a confusing bit of
logic unexplained, ask yourself: would this make sense to me in six
months? If the answer is no, it's probably worth the extra minute now.
Trust me, we've all paid that price before.
"""


if __name__ == "__main__":
    profiler = StyleProfiler()
    samples = {
        "casual_reddit": CASUAL_REDDIT_SAMPLE,
        "academic_abstract": ACADEMIC_ABSTRACT_SAMPLE,
        "conversational_blog": CONVERSATIONAL_BLOG_SAMPLE,
    }

    for name, text in samples.items():
        profile = profiler.profile([text])
        print(f"\n=== {name} ===")
        print(profile.to_json())
