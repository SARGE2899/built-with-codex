"""Learn different answers by changing weights using mean squared error.

Read read.md alongside this file. Only Python's standard library is used.
If you want to execute the example: python code.py

The ENTIRE training data is:
    'I drink' -> 'tea'
    'I eat'   -> 'rice'

Input features: [count of i, count of drink, count of eat].
Possible outputs: [tea, rice].
The target is [1, 0] for tea and [0, 1] for rice.

This is a tiny linear classifier, not an LLM. It has no hidden layer and
discards input word order. Its purpose is to expose the training arithmetic.
Unlike our earlier example, this file uses MEAN SQUARED ERROR, not cross-entropy.
"""

import math


INPUT_WORDS = ["i", "drink", "eat"]
OUTPUT_WORDS = ["tea", "rice"]
TRAINING_DATA = [("I drink", "tea"), ("I eat", "rice")]
LEARNING_RATE = 0.1
EPOCHS = 200

# Each row belongs to an OUTPUT; each column belongs to an INPUT feature.
#                  i  drink  eat
# Tea's weights:   1    0     2
# Rice's weights:  1    1     2
# These are chosen starting values from our worked example, not learned values.
INITIAL_WEIGHTS = [[1.0, 0.0, 2.0], [1.0, 1.0, 2.0]]
INITIAL_BIASES = [0.0, 0.0]


def encode(text):
    """Count input words in the fixed order [i, drink, eat]."""
    counts = [0.0] * len(INPUT_WORDS)
    for word in text.lower().split():
        if word not in INPUT_WORDS:
            raise ValueError(f"Unknown input word {word!r}; allowed: {INPUT_WORDS}")
        # .index(word) finds the position of this word in INPUT_WORDS.
        counts[INPUT_WORDS.index(word)] += 1.0
    return counts


def make_target(correct_word):
    """Give the correct output a target of 1; give all others a target of 0."""
    if correct_word not in OUTPUT_WORDS:
        raise ValueError(f"Unknown output word: {correct_word!r}")
    # Tea -> [1, 0]. Rice -> [0, 1]. These are targets, not input counts.
    return [1.0 if word == correct_word else 0.0 for word in OUTPUT_WORDS]


def forward(inputs, weights, biases):
    """Calculate scores and probabilities WITHOUT changing any weights."""
    scores = []
    for output_id in range(len(OUTPUT_WORDS)):
        score = biases[output_id]
        for input_id in range(len(INPUT_WORDS)):
            score += inputs[input_id] * weights[output_id][input_id]
        scores.append(score)

    # Softmax turns the two scores into probabilities that sum to 1.
    # Subtracting the largest score prevents overflow without changing results.
    largest = max(scores)
    exponentials = [math.exp(score - largest) for score in scores]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    return scores, probabilities


def mse(probabilities, target):
    """Average the squared differences across the OUTPUT choices."""
    squared_errors = [(p - t) ** 2 for p, t in zip(probabilities, target)]
    # The divisor is 2 because we have TWO OUTPUTS, not two training examples.
    return sum(squared_errors) / len(OUTPUT_WORDS)


def gradients(inputs, probabilities, target):
    """Differentiate MSE THROUGH softmax, then through the weighted sums.

    This is the chain rule: weight -> score -> probabilities -> loss.
    The familiar 'probability - target' shortcut for softmax + cross-entropy
    DOES NOT give the correct score gradient for our MSE loss.
    """
    output_count = len(OUTPUT_WORDS)

    # First: how does MSE change with each output probability?
    # dL/dp[k] = (2 / output_count) * (p[k] - target[k]).
    probability_gradients = [
        (2.0 / output_count) * (p - t)
        for p, t in zip(probabilities, target)
    ]

    score_gradients = []
    for j in range(output_count):
        score_gradient = 0.0
        for k in range(output_count):
            # Softmax links ALL probabilities to EACH score.
            # dp[k]/dz[j] = p[k] * (indicator(k == j) - p[j]).
            indicator = 1.0 if k == j else 0.0
            softmax_derivative = probabilities[k] * (indicator - probabilities[j])
            score_gradient += probability_gradients[k] * softmax_derivative
        score_gradients.append(score_gradient)

    # Since score[j] = sum(weight[j][i] * input[i]) + bias[j]:
    # dL/dweight[j][i] = dL/dscore[j] * input[i].
    # dL/dbias[j] = dL/dscore[j].
    weight_gradients = [
        [score_gradient * value for value in inputs]
        for score_gradient in score_gradients
    ]
    return weight_gradients, score_gradients


def train_one(inputs, target, weights, biases, learning_rate):
    """Use ONE labelled example to update the current model in place."""
    _, probabilities = forward(inputs, weights, biases)
    # Calculate all gradients from the SAME pre-update parameter state.
    weight_gradients, bias_gradients = gradients(inputs, probabilities, target)

    for output_id in range(len(OUTPUT_WORDS)):
        for input_id in range(len(INPUT_WORDS)):
            # new weight = old weight - learning rate * weight gradient.
            weights[output_id][input_id] -= (
                learning_rate * weight_gradients[output_id][input_id]
            )
        biases[output_id] -= learning_rate * bias_gradients[output_id]


def show_parameters(weights, biases):
    """Show which adjustable numbers belong to each candidate output."""
    for word, row, bias in zip(OUTPUT_WORDS, weights, biases):
        values = ", ".join(f"{value:.6f}" for value in row)
        print(f"  {word:4s}: weights=[{values}], bias={bias:.6f}")


def report(weights, biases):
    """Check BOTH examples without training on them during this function."""
    losses = []
    for text, correct_word in TRAINING_DATA:
        _, probabilities = forward(encode(text), weights, biases)
        loss = mse(probabilities, make_target(correct_word))
        losses.append(loss)
        # max(..., key=...) chooses the ID with the largest probability.
        # This is deterministic selection, not random probability sampling.
        choice = max(range(len(OUTPUT_WORDS)), key=lambda i: probabilities[i])
        print(f"  {text!r}: tea={probabilities[0]:.2%}, rice={probabilities[1]:.2%}"
              f" -> {OUTPUT_WORDS[choice]} | target={correct_word} | MSE={loss:.6f}")
    # This is a SECOND average, now across the TWO TRAINING EXAMPLES.
    print(f"  Mean loss across examples: {sum(losses) / len(losses):.6f}")


def main():
    # Copy each row so training does not change the documented initial values.
    weights = [row.copy() for row in INITIAL_WEIGHTS]
    biases = INITIAL_BIASES.copy()

    print("Input positions:", INPUT_WORDS)
    print("Possible outputs:", OUTPUT_WORDS)
    print("Training examples:", TRAINING_DATA)
    print("\nBEFORE TRAINING")
    show_parameters(weights, biases)
    report(weights, biases)

    # Trace the first update independently on a disposable model copy.
    # The main training model remains at its original starting values.
    demo_weights = [row.copy() for row in weights]
    demo_biases = biases.copy()
    inputs = encode("I drink")
    target = make_target("tea")
    scores, probabilities = forward(inputs, demo_weights, demo_biases)
    weight_gradients, bias_gradients = gradients(inputs, probabilities, target)
    print("\nONE WORKED UPDATE: 'I drink' -> tea")
    print("  Input:", inputs, "Target:", target)
    print("  Starting scores:", scores)
    print("  Starting probabilities:", probabilities)
    print("  Starting MSE:", mse(probabilities, target))
    print("  Score/bias gradients:", bias_gradients)
    print("  Weight gradients:", weight_gradients)
    train_one(inputs, target, demo_weights, demo_biases, LEARNING_RATE)
    show_parameters(demo_weights, demo_biases)
    _, after = forward(inputs, demo_weights, demo_biases)
    print(f"  After update: tea={after[0]:.2%}, rice={after[1]:.2%}, MSE={mse(after, target):.6f}")

    # One epoch means one pass through BOTH examples.
    # We update after each example: this is sequential / stochastic gradient
    # descent with a fixed example order, not a batch-average update.
    # 200 epochs * 2 examples = 400 updates to the main model.
    for epoch in range(1, EPOCHS + 1):
        for text, correct_word in TRAINING_DATA:
            train_one(encode(text), make_target(correct_word), weights, biases, LEARNING_RATE)
        if epoch in (1, 2, 5, 10, 50, 100, 200):
            print(f"\nAFTER EPOCH {epoch} (both updates completed)")
            report(weights, biases)

    print("\nFINAL PARAMETERS")
    show_parameters(weights, biases)
    print("\nThese results are on the training examples, not an unseen test set.")
    print("No weights are saved to disk. A fresh run starts from the initial values.")


if __name__ == "__main__":
    main()
