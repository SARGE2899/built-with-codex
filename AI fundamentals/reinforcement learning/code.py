"""Reinforcement learning: select a completion, then receive a reward.

This is a contextual bandit: one decision and immediate reward per episode.
It is a minimal RL setting, not multi-step planning or an LLM RLHF system.
The policy sees a context and chooses bread/sugar. A deliberately written toy
environment rewards matching the earlier clue. This is not an objective claim
about the only correct English completion.
Read read.md. Optional command: python code.py. Standard library only.
"""

import random

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


RATE = 0.2
ROUNDS = 100  # Each round visits both contexts: 200 one-action episodes.
SEED = 7


def sample_action(probabilities, rng):
    """Actually sample: exploration allows either action during training."""
    draw = rng.random()
    cumulative = 0.0
    for action_id, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return action_id
    return len(probabilities) - 1  # Guard floating-point rounding at the edge.


def environment_reward(context, action_id):
    """Toy environment feedback AFTER the policy selects an action.

    The environment encodes our desired behaviour. Reward design is a human
    choice here, not truth magically discovered by the model. The training
    policy receives only the reward for its chosen action, not a target vector.
    """
    x = encode(context)
    if x == [1.0, 0.0]:
        preferred_action = "bread"
    elif x == [0.0, 1.0]:
        preferred_action = "sugar"
    else:
        raise ValueError("This toy environment defines rewards only for the two lesson contexts.")
    return 1.0 if OUTPUTS[action_id] == preferred_action else -1.0


def policy_score_gradients(probabilities, action_id, reward):
    """Gradient of the sampled surrogate loss -reward * log P(chosen action).

    reward is held constant while differentiating this sampled action's loss.
    This is the basic score-function / REINFORCE estimator, with no baseline.
    It is NOT MSE and action_id is the selected action, not a correct label.
    """
    return [reward * (p - (1.0 if j == action_id else 0.0))
            for j, p in enumerate(probabilities)]


def train(rounds=ROUNDS, rate=RATE, verbose=False):
    weights, biases = new_model()
    rng = random.Random(SEED)
    rewards = []
    for round_number in range(1, rounds + 1):
        for context in CONTEXTS:
            x = encode(context)
            probabilities = predict(x, weights, biases)
            action_id = sample_action(probabilities, rng)
            reward = environment_reward(context, action_id)
            rewards.append(reward)
            g = policy_score_gradients(probabilities, action_id, reward)
            if verbose and len(rewards) == 1:
                print("FIRST EPISODE:", context)
                print("  Before:", probabilities, "Selected:", OUTPUTS[action_id], "Reward:", reward)
                print("  Score gradients:", g)
                print("  Sampled surrogate loss:", -reward * math.log(probabilities[action_id]))
            apply_update(x, g, weights, biases, rate)
            if verbose and len(rewards) == 1:
                print("  After:", predict(x, weights, biases))
        if verbose and round_number in (1, 10, 50, 100):
            print(f"After round {round_number}: episodes={len(rewards)}, cumulative mean reward={sum(rewards)/len(rewards):.4f}")
            show_predictions(weights, biases)
    return weights, biases, rewards


def main():
    print("REINFORCEMENT LEARNING: context -> sampled action -> reward -> update")
    print("Two actions:", OUTPUTS, "Rewards: match earlier clue +1; other action -1.")
    print("\nBEFORE TRAINING")
    show_predictions(*new_model())
    print("\nTRAINING")
    weights, biases, rewards = train(verbose=True)
    print("\nFINAL weights:", weights, "biases:", biases)
    print("Rewarded choices:", rewards.count(1.0), "Penalised choices:", rewards.count(-1.0))
    print("Sampling is used in training; final displayed choices use the maximum probability.")
    print("This learns our toy reward rule, not independently verified language correctness.")


if __name__ == "__main__":
    main()
