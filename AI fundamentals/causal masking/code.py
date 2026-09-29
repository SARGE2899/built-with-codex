"""Causal masking: prevent attention from looking at future token positions.

Run: python code.py
Read read.md for the theory and numerical calculations.
Only Python's standard library is required.

This demonstrates the MASK and attention softmax, not a trained language model.
Scores are supplied teaching values, not learned or computed from embeddings.
"""

import math


# Treat each word as one token for this small demonstration.
# Actual tokenizers may also use word fragments and punctuation.
TOKENS = ["I", "drink", "tea", "with", "sugar"]

# These are the same illustrative attention scores from the explanation.
# For the main example, they describe the query at the 'drink' position.
# Treat them as ALREADY SCALED scores: do not divide by sqrt(d_k) again.
SCORES = [1.0, 2.0, 5.0, 3.0, 4.0]
QUERY_POSITION = 1  # Python starts counting at 0: I=0, drink=1, tea=2, ...


def causal_mask(length):
    """Return allowed[q][k]: may query position q use key position k?

    A position can attend to itself and earlier positions: k <= q.
    True means allowed in THIS demonstration. Library mask conventions vary.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("length must be a positive integer.")
    return [[key_position <= query_position for key_position in range(length)]
            for query_position in range(length)]


def masked_softmax(scores, allowed):
    """Mask forbidden scores BEFORE converting scores into probabilities."""
    if not scores or len(scores) != len(allowed):
        raise ValueError("Provide equally sized, nonempty scores and mask lists.")
    if not all(isinstance(flag, bool) for flag in allowed):
        raise ValueError("Mask entries must be True or False.")
    if not any(allowed):
        # Softmax needs at least one permitted position. Otherwise every
        # exponential is zero and normalization would divide by zero.
        raise ValueError("At least one position must be allowed.")
    if not all(math.isfinite(score) for score in scores):
        raise ValueError("Supply finite scores; this function applies the infinities.")

    # Negative infinity makes exp(score) exactly zero at forbidden positions.
    masked_scores = [score if keep else -math.inf
                     for score, keep in zip(scores, allowed)]

    # Stable softmax subtracts the largest PERMITTED score from every score.
    # This changes neither the mathematical probabilities nor the zero mask.
    highest = max(masked_scores)
    exponentials = [math.exp(score - highest) for score in masked_scores]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    return masked_scores, probabilities


def main():
    mask = causal_mask(len(TOKENS))
    print("CAUSAL MASK: rows are query positions; columns are key positions")
    print("YES means the column's information is available to that row.")
    print(f"{'query / key':>12}" + "".join(f"{word:>9}" for word in TOKENS))
    for position, row in enumerate(mask):
        print(f"{TOKENS[position]:>12}" + "".join(
            f"{'YES' if allowed else 'NO':>9}" for allowed in row))

    print("\nSHIFTED NEXT-WORD TARGETS from the same sentence:")
    # The target is the word AFTER the last visible input word.
    # No target after 'sugar' was supplied, so don't invent an end token here.
    for position in range(len(TOKENS) - 1):
        prefix = " ".join(TOKENS[:position + 1])
        target = TOKENS[position + 1]
        print(f"  Input: {prefix!r}; target for loss calculation: {target!r}")

    # Compare unrestricted attention with causally masked attention for drink.
    # The all-True row below is an intentionally unsafe comparison for this
    # next-token training setting; it permits seeing future target information.
    _, unmasked = masked_softmax(SCORES, [True] * len(TOKENS))
    masked_scores, attention = masked_softmax(SCORES, mask[QUERY_POSITION])
    print("\nWORKED EXAMPLE: query at 'drink'; next-word target is 'tea'")
    print("Scores are supplied illustrative, already-scaled attention scores.")
    print(f"{'position':>10} {'raw score':>12} {'masked score':>14} {'unmasked':>12} {'masked':>12}")
    for word, raw, masked, before, after in zip(
            TOKENS, SCORES, masked_scores, unmasked, attention):
        print(f"{word:>10} {raw:>12.1f} {masked:>14.1f} {before:>11.4%} {after:>11.4%}")

    # A very large forbidden score still has ZERO attention after masking.
    # This is an explicit numerical experiment, not extra training data.
    changed_scores = SCORES.copy()
    changed_scores[2] = 1_000_000.0
    _, changed_attention = masked_softmax(changed_scores, mask[QUERY_POSITION])
    print("\nSet the forbidden tea score to 1,000,000:")
    print("Attention remains:", [round(p, 6) for p in changed_attention])

    print("\nThese probabilities describe attention to INPUT POSITIONS.")
    print("They are NOT probabilities of selecting a next output word.")
    print("Masking future tea as an INPUT does not forbid predicting tea as OUTPUT.")
    print("A full model would mix values, process representations, and calculate")
    print("vocabulary probabilities before comparing its prediction with tea.")
    print("No model training or weight updates occur in this demonstration.")


if __name__ == "__main__":
    main()
