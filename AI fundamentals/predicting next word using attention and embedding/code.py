"""A tiny, fully trainable attention example. Read read.md alongside this file.

Input: I had bread this morning and I drink tea with _____
Assumed training answer: bread (explicitly supplied, never inferred as truth).
Only ONE labelled example is used. No other training sentences are hidden here.

This demonstrates numerical mechanics, NOT general language understanding or
proof that attention discovers the right earlier word. With one target, the
model can learn to favour bread even when the input offers no relevant clue.

Only the Python standard library is required. Optional command: python code.py
"""

import math
import random
import re


PROMPT = "I had bread this morning and I drink tea with _____"
TARGET = "bread"
REQUESTED_VOCABULARY = ["i", "drink", "tea", "bread", "coffee", "sugar"]
DIM = 2                 # Two numbers in each embedding, query, key, and value.
LEARNING_RATE = 0.3
TRAINING_STEPS = 2000   # Repetitions of the ONE supplied example.
SEED = 7               # Reproducible starting weights, not a source of knowledge.


def tokenize(text):
    """Use lowercase whole words; underscores/punctuation are not tokens here."""
    return re.findall(r"[a-z]+", text.lower())


TOKENS = tokenize(PROMPT)
VOCABULARY = REQUESTED_VOCABULARY.copy()
for word in TOKENS:
    if word not in VOCABULARY:
        VOCABULARY.append(word)
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def matvec(matrix, vector):
    """Each matrix row gives one output: sum(row[i] * vector[i])."""
    return [dot(row, vector) for row in matrix]


def transpose_matvec(matrix, vector):
    """Send a gradient backwards through a matrix multiplication."""
    return [sum(matrix[j][i] * vector[j] for j in range(len(matrix)))
            for i in range(len(matrix[0]))]


def softmax(scores):
    largest = max(scores)
    terms = [math.exp(value - largest) for value in scores]
    return [value / sum(terms) for value in terms]


def softmax_backward(probabilities, upstream):
    """Chain rule through softmax: works for attention AND output softmax.

    upstream[j] means the derivative of loss with respect to probability j.
    Changing one score affects every probability, not just its own.
    """
    common = dot(probabilities, upstream)
    return [p * (g - common) for p, g in zip(probabilities, upstream)]


def new_model():
    rng = random.Random(SEED)

    def matrix(rows, columns):
        return [[rng.uniform(-0.5, 0.5) for _ in range(columns)] for _ in range(rows)]

    return {
        "embedding": matrix(len(VOCABULARY), DIM),
        "query": matrix(DIM, DIM),
        "key": matrix(DIM, DIM),
        "value": matrix(DIM, DIM),
        "output": matrix(len(VOCABULARY), DIM),
        "bias": [0.0] * len(VOCABULARY),
    }


def forward(model, text):
    """The target is NOT an argument. Prediction only sees text and parameters."""
    words = tokenize(text)
    if not words:
        raise ValueError("Supply at least one known word.")
    unknown = [word for word in words if word not in WORD_TO_ID]
    if unknown:
        raise ValueError(f"Unknown words: {unknown}. Vocabulary: {VOCABULARY}")
    ids = [WORD_TO_ID[word] for word in words]

    # Embedding lookup: select a word's learned row, not a count vector.
    # Add a small fixed position signal so repeated 'i' tokens can differ.
    # This simple [position/20, -position/20] rule is illustrative, not the
    # positional encoding used by all real LLMs. It is not trained here.
    inputs = []
    for position, word_id in enumerate(ids):
        e = model["embedding"][word_id]
        offset = position / 20.0
        inputs.append([e[0] + offset, e[1] - offset])

    # Predict the next token from the FINAL input position ('with').
    # Its query compares with keys for ALL existing positions, including itself.
    # We do not feed the withheld target into these calculations.
    query = matvec(model["query"], inputs[-1])
    keys = [matvec(model["key"], x) for x in inputs]
    values = [matvec(model["value"], x) for x in inputs]
    scores = [dot(query, key) / math.sqrt(DIM) for key in keys]
    attention = softmax(scores)  # Distribution across INPUT POSITIONS.
    context = [sum(a * v[j] for a, v in zip(attention, values)) for j in range(DIM)]

    logits = [dot(row, context) + b for row, b in zip(model["output"], model["bias"])]
    probabilities = softmax(logits)  # Distribution across OUTPUT WORDS.
    # Save intermediate results so backpropagation can reuse the same forward pass.
    return dict(words=words, ids=ids, inputs=inputs, query=query, keys=keys,
                values=values, scores=scores, attention=attention, context=context,
                logits=logits, probabilities=probabilities)


def loss(probabilities, target_id):
    """MSE over all 11 output words, preserving the previous lesson's loss."""
    return sum((p - (1.0 if j == target_id else 0.0)) ** 2
               for j, p in enumerate(probabilities)) / len(probabilities)


def backward(model, cache, target_id):
    """Calculate ALL gradients before changing ANY parameters.

    Path: MSE -> output softmax -> output weights -> context mixture ->
    attention softmax -> Q/K/V transforms -> input embedding rows.
    """
    gradients = {name: ([[0.0] * len(row) for row in parameter] if name != "bias"
                        else [0.0] * len(parameter)) for name, parameter in model.items()}
    p = cache["probabilities"]
    dp = [(2.0 / len(p)) * (value - (1.0 if j == target_id else 0.0))
          for j, value in enumerate(p)]
    dz = softmax_backward(p, dp)
    gradients["bias"] = dz.copy()
    gradients["output"] = [[g * c for c in cache["context"]] for g in dz]
    dc = transpose_matvec(model["output"], dz)

    # context = sum(attention[i] * value[i]). Both factors receive gradients.
    da = [dot(dc, value) for value in cache["values"]]
    dv = [[a * g for g in dc] for a in cache["attention"]]
    ds = softmax_backward(cache["attention"], da)
    scale = math.sqrt(DIM)
    dq = [sum(g * key[j] / scale for g, key in zip(ds, cache["keys"])) for j in range(DIM)]
    dk = [[g * q / scale for q in cache["query"]] for g in ds]

    def back_transform(name, x, upstream):
        # For y = W*x, dW[row][col] = upstream[row]*x[col].
        for row in range(DIM):
            for col in range(DIM):
                gradients[name][row][col] += upstream[row] * x[col]
        return transpose_matvec(model[name], upstream)

    dx = [[0.0] * DIM for _ in cache["inputs"]]
    for i, x in enumerate(cache["inputs"]):
        from_key = back_transform("key", x, dk[i])
        from_value = back_transform("value", x, dv[i])
        dx[i] = [a + b for a, b in zip(from_key, from_value)]
    from_query = back_transform("query", cache["inputs"][-1], dq)
    dx[-1] = [a + b for a, b in zip(dx[-1], from_query)]

    # The same word can appear at multiple positions. Add their gradients
    # into the SAME embedding row; do not overwrite one with the other.
    for word_id, input_gradient in zip(cache["ids"], dx):
        for j in range(DIM):
            gradients["embedding"][word_id][j] += input_gradient[j]
    return gradients


def update(model, gradients):
    for name, parameter in model.items():
        if name == "bias":
            for i in range(len(parameter)):
                parameter[i] -= LEARNING_RATE * gradients[name][i]
        else:
            for i, row in enumerate(parameter):
                for j in range(len(row)):
                    row[j] -= LEARNING_RATE * gradients[name][i][j]


def show_forward(model, text):
    cache = forward(model, text)
    print("\nInput:", text)
    print("Query at the final input position:", [round(x, 6) for x in cache["query"]])
    print("Position | word    | embedding + position | key | value | score | attention")
    for i, word in enumerate(cache["words"]):
        lists = [str([round(v, 4) for v in cache[name][i]]) for name in ("inputs", "keys", "values")]
        print(f"{i:8d} | {word:7s} | {' | '.join(lists)} | {cache['scores'][i]:.6f} | {cache['attention'][i]:.2%}")
    print("Attention mixture:", [round(x, 6) for x in cache["context"]])
    print("\nALL possible next words (these are output probabilities, not attention):")
    for word, score, p in zip(VOCABULARY, cache["logits"], cache["probabilities"]):
        print(f"  {word:7s} score={score: .6f} probability={p:.4%}")
    best = max(range(len(VOCABULARY)), key=lambda j: cache["probabilities"][j])
    print("Selected next word:", VOCABULARY[best])
    return cache


def main():
    model = new_model()
    target_id = WORD_TO_ID[TARGET]
    print("Vocabulary:", VOCABULARY)
    print("Tokens:", TOKENS)
    print("Only labelled example:", " ".join(TOKENS), "->", TARGET)
    print("Bread is an assumed training target, not the uniquely correct continuation.")
    print("\nBEFORE TRAINING")
    show_forward(model, PROMPT)
    print("\nAUTOMATIC TRAINING (metrics printed AFTER each indicated update)")
    for step in range(1, TRAINING_STEPS + 1):
        cache = forward(model, PROMPT)
        gradients = backward(model, cache, target_id)
        update(model, gradients)
        if step in (1, 10, 100, 500, 1000, 2000):
            p = forward(model, PROMPT)["probabilities"]
            print(f"  Step {step:4d}: P(bread)={p[target_id]:.4%}, MSE={loss(p, target_id):.8f}")
    print("\nAFTER TRAINING")
    show_forward(model, PROMPT)
    print("\nDiagnostic: a prediction-only probe; this is NOT extra training data.")
    probe = "I drink tea with"
    p = forward(model, probe)["probabilities"]
    print(f"  {probe!r}, no earlier bread: P(bread)={p[target_id]:.4%}")
    print("One repeated target can teach a bread preference even without the clue.")
    print("Attention is not guaranteed to focus on bread or reveal a causal explanation.")
    print("No general language skill is established; no model file is saved.")


if __name__ == "__main__":
    main()
