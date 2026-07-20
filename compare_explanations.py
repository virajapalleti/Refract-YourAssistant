import json
import time

from explanation_generator import ExplanationGenerator, load_eval_profiles

TEST_CONCEPTS = [
    {
        "concept": "hash table collision resolution",
        "context": (
            "A hash table stores key-value pairs by computing a hash function on each key "
            "to determine which bucket the pair belongs in. When two different keys produce "
            "the same hash value, a collision occurs. Common resolution strategies include "
            "chaining, where each bucket holds a linked list of entries, and open addressing, "
            "where the algorithm probes subsequent slots until an empty one is found. The "
            "choice of collision resolution strategy significantly affects lookup performance, "
            "especially as the load factor increases."
        ),
    },
    {
        "concept": "CRISPR gene editing mechanism",
        "context": (
            "CRISPR-Cas9 is a molecular tool adapted from a bacterial immune defense system. "
            "It uses a guide RNA sequence to direct the Cas9 enzyme to a specific location in "
            "an organism's DNA. Once bound, Cas9 cuts both strands of the double helix at the "
            "target site. The cell's natural repair machinery then either disables the gene "
            "through imprecise repair or inserts a new sequence provided by researchers. This "
            "technology enables precise, targeted modifications to virtually any genome."
        ),
    },
    {
        "concept": "compound interest",
        "context": (
            "Compound interest is the process by which interest earned on a principal amount "
            "is reinvested, so that in subsequent periods, interest is calculated on both the "
            "original principal and the accumulated interest. The formula A = P(1 + r/n)^(nt) "
            "captures this growth, where P is the principal, r is the annual rate, n is the "
            "compounding frequency, and t is the time in years. Over long horizons, compounding "
            "produces exponential growth, which is why early and consistent investing is "
            "emphasized in personal finance."
        ),
    },
]


def main():
    profiles = load_eval_profiles()
    generator = ExplanationGenerator()
    results = {}

    for item in TEST_CONCEPTS:
        concept = item["concept"]
        context = item["context"]
        results[concept] = {}

        print(f"\n{'#' * 70}")
        print(f"  CONCEPT: {concept}")
        print(f"{'#' * 70}")

        print(f"\n{'=' * 60}")
        print("BASELINE (no style adaptation)")
        print(f"{'=' * 60}")
        baseline = generator.generate_baseline(concept, context)
        print(baseline)
        results[concept]["baseline"] = baseline
        time.sleep(3)

        for profile_name, profile in profiles.items():
            print(f"\n{'=' * 60}")
            print(f"PROFILE: {profile_name}")
            print(f"{'=' * 60}")
            explanation = generator.generate(concept, context, profile)
            print(explanation)
            results[concept][profile_name] = explanation
            time.sleep(1)

    with open("compare_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'#' * 70}")
    print("Results saved to compare_results.json")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
