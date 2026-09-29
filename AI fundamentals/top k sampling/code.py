"""Top-k sampling: choose only among the k most likely next words.

Read read.md alongside this file. Run with: python code.py
Uses Python's standard library only. No model download or training is needed.
The probabilities below are our illustrative example, NOT a model's output.
"""

import math
import random
from collections import Counter


# Keep the exact example from the lesson. Each word stands for one token here.
# A real language model can have tokens smaller than complete words.
PROMPT = "I drink"

# A dictionary connects a word to its probability.
# Decimal 0.70 means 70%. These three numbers add to 1 (100%).
# We supply them directly so we can focus on selection after prediction.
PROBABILITIES = {"tea": 0.70, "coffee": 0.25, "sugar": 0.05}
K = 2
SEED = 7
TRIALS = 10_000  # The underscore makes the number easier to read: 10000.


def top_k_distribution(probabilities, k):
    """Return a NEW dictionary containing only the retained words.

    1. Rank words by probability, highest first.
    2. Keep exactly k entries.
    3. Divide each retained probability by the retained total.

    The original dictionary remains unchanged. Exclusion from one selection
    does not erase a word from a model's vocabulary.
    """
    if not probabilities:
        raise ValueError("Provide at least one candidate word.")
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(probabilities):
        raise ValueError("k must be an integer between 1 and the candidate count.")
    if any(not math.isfinite(p) or p < 0 for p in probabilities.values()):
        raise ValueError("Probabilities must be finite and nonnegative.")
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("The input probabilities must add to 1.")

    # .items() gives pairs such as ('tea', 0.70).
    # pair[1] means the probability part of each pair.
    # reverse=True puts the largest probability first.
    # Equal probabilities keep their original dictionary order in this demo.
    ranked = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)

    # [:k] takes the first k pairs. With k=2: tea and coffee.
    retained = ranked[:k]
    retained_total = sum(probability for word, probability in retained)

    # For k=2, the retained total is 0.70 + 0.25 = 0.95.
    # tea: 0.70/0.95 = 0.736842...; coffee: 0.25/0.95 = 0.263158...
    # Their new probabilities add to 1, so we can sample between them.
    return {word: probability / retained_total for word, probability in retained}


def sample_word(distribution, rng):
    """Draw from the normalized distribution returned above.

    rng is a random-number generator passed in by the caller.
    Crucially, it receives ONLY retained candidates, never the removed ones.
    """
    # rng.random() gives a number from 0 (included) to 1 (excluded).
    draw = rng.random()
    cumulative = 0.0
    last_positive_word = None

    # For k=2, tea occupies [0, 0.736842...).
    # Coffee occupies [0.736842..., 1). Sugar has no interval at all.
    for word, probability in distribution.items():
        if probability > 0:
            last_positive_word = word
        cumulative += probability
        if draw < cumulative:
            return word

    # Floating-point rounding may make the cumulative sum slightly below 1.
    # In that edge case return the last retained word with positive probability.
    # This fallback cannot reintroduce sugar when sugar was filtered out.
    return last_positive_word


def main():
    print("TOP-K SAMPLING: one next-word selection, no training")
    print("Input:", PROMPT)
    print("Illustrative probabilities supplied directly:", PROBABILITIES)

    filtered = top_k_distribution(PROBABILITIES, K)
    print(f"\nAfter keeping the top {K} candidates:")
    for word in PROBABILITIES:
        # .get(word, 0.0) returns zero if this word is absent from the dictionary.
        probability = filtered.get(word, 0.0)
        print(f"  {word:6s}: {probability:.4%}")

    # A fixed seed makes this teaching run reproducible. It does not train
    # the model or determine which words top-k retains.
    selected = sample_word(filtered, random.Random(SEED))
    print("\nOne sampled completion:", PROMPT, selected)

    # These trials repeat the SAME selection with the SAME probabilities.
    # They do not build a sentence or calculate what follows tea or coffee.
    rng = random.Random(SEED)
    counts = Counter(sample_word(filtered, rng) for _ in range(TRIALS))
    print(f"\nRepeat this same next-word selection {TRIALS:,} times:")
    for word in PROBABILITIES:
        print(f"  {word:6s}: {counts[word]:5d} selections ({counts[word]/TRIALS:.2%})")
    print("The observed frequencies need not exactly equal the probabilities.")
    print("Sugar cannot be selected here because it is not a sampling candidate.")

    print("\nCompare k settings using the SAME original probabilities:")
    for k in (1, 2, 3):
        distribution = top_k_distribution(PROBABILITIES, k)
        print(f"  k={k}:", {word: f"{p:.4%}" for word, p in distribution.items()})

    print("\nTop-k affects inference selection. No weights changed.")
    print("At a later generation step, a model would compute fresh probabilities.")


# Execute the demonstration only when this file is run directly.
if __name__ == "__main__":
    main()
