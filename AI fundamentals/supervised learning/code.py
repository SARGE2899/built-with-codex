"""Supervised learning: people supply input/answer pairs.

Read read.md for the actual numbers. Optional command: python code.py
No packages, downloaded model, internet, or saved model file are required.
This is a tiny labelled classification demonstration, not general language AI.
"""

import math
import re


# We deliberately use the SAME two contexts throughout these four lessons.
# The sugar context is an explicitly added comparison to the original bread one.
CONTEXTS = [
    "I had bread this morning and I drink tea with",
    "I had sugar this morning and I drink tea with",
]
OUTPUTS = ["bread", "sugar"]
FEATURES = ["bread", "sugar"]


def encode(text):
    """Count the two clue words in the INPUT, not in the desired output.

    bread context -> [1, 0]; sugar context -> [0, 1].
    This hand-designed representation ignores the other words and their order.
    It is NOT an embedding, attention, or a full language model. We simplify
    representation to isolate the difference between learning signals.
    """
    words = re.findall(r"[a-z]+", text.lower())
    return [float(words.count(word)) for word in FEATURES]


def new_model():
    # One output row per candidate, one weight per input feature.
    #              input bread   input sugar
    # output bread      0             0
    # output sugar      0             0
    # Biases are separate adjustable additions to each output's score.
    return [[0.0, 0.0], [0.0, 0.0]], [0.0, 0.0]


def predict(x, weights, biases):
    """Weighted sums -> softmax. This function never sees an answer or reward."""
    scores = [sum(w * value for w, value in zip(row, x)) + bias
              for row, bias in zip(weights, biases)]
    # Subtracting the maximum keeps exponentials numerically manageable.
    terms = [math.exp(score - max(scores)) for score in scores]
    return [term / sum(terms) for term in terms]


def apply_update(x, score_gradients, weights, biases, rate):
    """Gradient descent: new parameter = old parameter - rate * gradient."""
    # All supplied gradients were calculated BEFORE changing any parameters.
    for output_id in range(len(OUTPUTS)):
        for feature_id in range(len(FEATURES)):
            gradient = score_gradients[output_id] * x[feature_id]
            weights[output_id][feature_id] -= rate * gradient
        biases[output_id] -= rate * score_gradients[output_id]


def show_predictions(weights, biases):
    for text in CONTEXTS:
        p = predict(encode(text), weights, biases)
        best = max(range(len(OUTPUTS)), key=lambda j: p[j])
        print(f"  {text!r} -> bread={p[0]:.2%}, sugar={p[1]:.2%}; selected={OUTPUTS[best]}")
    # Highest-probability selection is used for display. At a tie, the first
    # output wins by list order; this is not a learned preference.


def mse(probabilities, target_id):
    # Mean across TWO OUTPUTS, not across the number of training examples.
    return sum((p - (1.0 if j == target_id else 0.0)) ** 2
               for j, p in enumerate(probabilities)) / len(probabilities)


def mse_score_gradients(probabilities, target_id):
    """Differentiate MSE through softmax; this is NOT the CE shortcut."""
    # First find dL/dp. Then account for how each score affects every probability.
    dp = [(2.0 / len(probabilities)) * (p - (1.0 if j == target_id else 0.0))
          for j, p in enumerate(probabilities)]
    common = sum(p * g for p, g in zip(probabilities, dp))
    return [p * (g - common) for p, g in zip(probabilities, dp)]


def train(pairs, epochs=100, rate=0.2, verbose=False):
    weights, biases = new_model()
    # This is the same training function in the supervised and self-supervised
    # files. Only the source of the target labels differs between those files.
    for epoch in range(1, epochs + 1):
        for text, correct_word in pairs:
            x = encode(text)
            target_id = OUTPUTS.index(correct_word)
            p = predict(x, weights, biases)
            g = mse_score_gradients(p, target_id)
            apply_update(x, g, weights, biases, rate)
        if verbose and epoch in (1, 10, 50, 100):
            average = sum(mse(predict(encode(text), weights, biases), OUTPUTS.index(y))
                          for text, y in pairs) / len(pairs)
            print(f"After epoch {epoch}: mean training MSE={average:.6f}")
            show_predictions(weights, biases)
    return weights, biases


# These are deliberately supplied labels, NOT facts deduced from the sentences.
# Bread/sugar are chosen desired completions, not uniquely correct English answers.
LABELLED_EXAMPLES = [(CONTEXTS[0], "bread"), (CONTEXTS[1], "sugar")]


def main():
    print("SUPERVISED: a person supplied each desired answer.")
    for text, label in LABELLED_EXAMPLES:
        print("Input:", text, "Features:", encode(text), "Label:", label)
    print("\nBEFORE TRAINING")
    show_predictions(*new_model())
    print("\nTRAINING: 100 passes x 2 examples = 200 updates")
    weights, biases = train(LABELLED_EXAMPLES, verbose=True)
    print("\nFINAL weights:", weights, "biases:", biases)
    print("These are training-set results. No unseen test performance is established.")


if __name__ == "__main__":
    main()
