"""
Validates that StyleProfiler produces measurably different profiles across
the eval_profiles/ writing samples. Prints a comparison table and a
differentiation check, then saves all profiles to eval_style_profiles.json.
"""

import json
import os
import re
import statistics

from style_profiler import StyleProfiler

PROFILE_DIR = "eval_profiles"
PROFILE_FILES = [
    "profile_1_concise_technical.txt",
    "profile_2_verbose_explainer.txt",
    "profile_3_structured_notetaker.txt",
    "profile_4_casual_conversational.txt",
    "profile_5_academic_formal.txt",
]

NUMERIC_FIELDS = [
    "avg_sentence_length",
    "vocabulary_richness",
    "flesch_kincaid_grade",
    "formality_score",
    "avg_paragraph_length",
    "jargon_comfort",
]
BOOLEAN_FIELDS = ["uses_analogies", "uses_questions"]

SAMPLE_HEADER_RE = re.compile(r"SAMPLE \d+:\s*")

CV_THRESHOLD_PCT = 15


def load_samples(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = SAMPLE_HEADER_RE.split(content)
    return [p.strip() for p in parts if p.strip()]


def print_table(profiles: dict):
    columns = NUMERIC_FIELDS + BOOLEAN_FIELDS
    header = f"{'file':38s}" + "".join(f"{c:>22s}" for c in columns)
    print(header)
    print("-" * len(header))
    for filename, profile in profiles.items():
        d = profile.to_dict()
        row = f"{filename:38s}" + "".join(f"{str(d[c]):>22s}" for c in columns)
        print(row)


def check_differentiation(profiles: dict) -> bool:
    print("\nDifferentiation check:")
    all_passed = True

    for field in NUMERIC_FIELDS:
        values = [profile.to_dict()[field] for profile in profiles.values()]
        mean = statistics.mean(values)
        stdev = statistics.pstdev(values)
        cv = (stdev / mean * 100) if mean else 0.0
        passed = cv > CV_THRESHOLD_PCT
        all_passed = all_passed and passed
        print(f"  {field:24s} CV={cv:6.1f}%  {'PASS' if passed else 'FAIL'}")

    for field in BOOLEAN_FIELDS:
        values = {profile.to_dict()[field] for profile in profiles.values()}
        passed = len(values) > 1
        all_passed = all_passed and passed
        print(f"  {field:24s} unique={len(values)}     {'PASS' if passed else 'FAIL'}")

    return all_passed


def main():
    profiler = StyleProfiler()
    profiles = {}

    for filename in PROFILE_FILES:
        path = os.path.join(PROFILE_DIR, filename)
        samples = load_samples(path)
        profiles[filename] = profiler.profile(samples)

    print_table(profiles)
    all_passed = check_differentiation(profiles)
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")

    output = {filename: profile.to_dict() for filename, profile in profiles.items()}
    with open("eval_style_profiles.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print("\nSaved eval_style_profiles.json")


if __name__ == "__main__":
    main()
