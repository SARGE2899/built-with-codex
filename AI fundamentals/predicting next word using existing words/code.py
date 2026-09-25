"""Our first tiny text model: learn to predict 'tea' after 'I drink'.

Run this file with:
    python code.py

Only Python's standard library is needed. No packages or internet are needed.

IMPORTANT DISTINCTION:
    ['i', 'drink', 'tea', 'coffee'] is our VOCABULARY: allowed words.
    'i drink tea' is our TRAINING SENTENCE: an example of word order.

A list of allowed words does not tell a model what follows 'i drink'.
We use the sentence from our conversation to supply that answer.
Coffee is allowed as an output, but we have not supplied an example teaching
the model to predict coffee. We have not added any other training sentences.

This is a tiny trainable next-word classifier, NOT a full LLM.
It shows input numbers -> weighted scores -> probabilities -> weight updates.
It learns one example; this does not demonstrate understanding or an ability
to handle new sentences.
"""

import math


# STEP 1: List every word our tiny model is allowed to use.
# A real tokenizer can use pieces of words. We use whole words for simplicity.
VOCABULARY = ["i", "drink", "tea", "coffee"]

# An ID is just a position in the list, not a measurement of meaning.
# This creates: {'i': 0, 'drink': 1, 'tea': 2, 'coffee': 3}.
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}

# STEP 2: Explicitly define the training example.
# We withhold the last word and use it as the correct answer.
TRAINING_SENTENCE = "i drink tea"
training_words = TRAINING_SENTENCE.split()
training_input = " ".join(training_words[:-1])  # 'i drink'
correct_next_word = training_words[-1]          # 'tea'
correct_word_id = WORD_TO_ID[correct_next_word]


def encode(text):
    """Represent text by counting occurrences of each vocabulary word."""
    # Start with four zeros, one slot per word: [i, drink, tea, coffee].
    numbers = [0.0] * len(VOCABULARY)

    # Lowercase makes 'I drink' and 'i drink' the same input here.
    for word in text.lower().split():
        if word not in WORD_TO_ID:
            raise ValueError(f"Unknown word: {word!r}. Allowed: {VOCABULARY}")
        numbers[WORD_TO_ID[word]] += 1.0

    # 'I drink' becomes [1, 1, 0, 0]. These are counts, NOT embeddings.
    # Limitation: counts lose word order. 'drink I' gets the same numbers.
    # Real LLMs use a more sophisticated representation that handles order.
    return numbers


# STEP 3: Create adjustable weights and biases.
# Each candidate output word has its own row of four input weights.
# There are four candidates, so this gives us 4 x 4 = 16 weights.
weights = [[0.0] * len(VOCABULARY) for _ in VOCABULARY]
biases = [0.0] * len(VOCABULARY)

# We start at zero to make the example easy to follow: all words initially tie.
# Zero initialization works for this simple linear classifier. It is generally
# unsuitable for all the weights in a multi-layer neural network.


def predict_probabilities(input_numbers):
    """Calculate one score per word, then apply softmax to those scores."""
    scores = []

    for output_id in range(len(VOCABULARY)):
        # This is the familiar equation:
        # score = x1*w1 + x2*w2 + x3*w3 + x4*w4 + bias
        score = biases[output_id]
        for input_id in range(len(VOCABULARY)):
            score += input_numbers[input_id] * weights[output_id][input_id]
        scores.append(score)

    # Softmax: probability of a word = exp(its score) / sum(exp(all scores)).
    # Subtracting the largest score prevents numerical overflow.
    # It does not change the mathematical probabilities.
    largest_score = max(scores)
    exponentials = [math.exp(score - largest_score) for score in scores]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def show_probabilities(probabilities):
    """Print all candidates so we can see what changed during training."""
    for word, probability in zip(VOCABULARY, probabilities):
        print(f"  {word:6s}: {probability:7.2%}")


def main():
    # STEP 4: Turn our training input into numbers.
    input_numbers = encode(training_input)
    print("Vocabulary:", VOCABULARY)
    print("Only training sentence:", TRAINING_SENTENCE)
    print(f"Training example: {training_input!r} -> {correct_next_word!r}")
    print("Input numbers [i, drink, tea, coffee]:", input_numbers)
    print("\nBEFORE TRAINING: every word has a 25% probability.")
    show_probabilities(predict_probabilities(input_numbers))

    # The learning rate controls the size of each adjustment.
    learning_rate = 0.1

    # We revisit the SAME example 200 times. This is not 200 new sentences.
    training_steps = 200
    print("\nTRAINING: the loss below is measured before each update.")

    for step in range(1, training_steps + 1):
        # A. Make a prediction with the current weights.
        probabilities = predict_probabilities(input_numbers)

        # B. Measure the mistake using cross-entropy for the correct word.
        # loss = -log(probability assigned to 'tea').
        # If P(tea) is near 1, loss is near 0. Lower loss is better.
        # max(..., 1e-15) protects against trying to calculate log(0).
        loss = -math.log(max(probabilities[correct_word_id], 1e-15))

        if step in (1, 2, 5, 10, 50, 100, 200):
            print(f"  Step {step:3d}: P(tea) = {probabilities[correct_word_id]:7.2%}, loss = {loss:.4f}")

        # C. Adjust every candidate's weights and bias.
        for output_id in range(len(VOCABULARY)):
            # Our desired probabilities are [0, 0, 1, 0]: tea is correct.
            target = 1.0 if output_id == correct_word_id else 0.0

            # For softmax + cross-entropy, the derivative with respect to
            # this candidate's score simplifies to probability - target.
            # This derivative is the direction information for training.
            # It is NOT the same thing as the loss printed above.
            score_gradient = probabilities[output_id] - target

            for input_id in range(len(VOCABULARY)):
                # The weight's gradient also depends on its input value.
                weight_gradient = score_gradient * input_numbers[input_id]

                # Gradient descent: new weight = old weight - rate*gradient.
                # For tea, probability - 1 is negative, so active weights rise.
                # For other words, probability - 0 is positive, so they fall.
                weights[output_id][input_id] -= learning_rate * weight_gradient

            # The bias is adjusted too. Its gradient is score_gradient.
            biases[output_id] -= learning_rate * score_gradient

    # STEP 5: Use the trained weights with the user's requested input.
    # No weight updates happen in this prediction step.
    user_input = "I drink"
    final_probabilities = predict_probabilities(encode(user_input))
    print(f"\nAFTER TRAINING -- input: {user_input!r}")
    show_probabilities(final_probabilities)

    # Choose the highest-probability word. We are not sampling randomly here.
    predicted_id = max(range(len(VOCABULARY)), key=lambda i: final_probabilities[i])
    predicted_word = VOCABULARY[predicted_id]
    print("\nPredicted next word:", predicted_word)
    print("Completed text:", user_input + " " + predicted_word)

    # We stop after ONE prediction. We have not taught this model what follows
    # 'i drink tea', so repeatedly extending the sentence would be misleading.
    print("\nTea won because it was the target in our only training example.")
    print("Coffee was an allowed candidate, but never a correct training answer.")
    print("This learned one example; it has not learned general language.")


if __name__ == "__main__":
    main()
