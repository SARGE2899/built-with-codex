"""Unsupervised learning: group the same input contexts without answer labels.

This script intentionally DOES NOT predict a next word. K-means discovers groups
under a chosen distance objective; it does not receive bread/sugar output targets.
The two word-count features are human-designed, so do not interpret these tiny
clusters as independently discovered semantic concepts.
Read read.md. Optional command: python code.py. Standard library only.
"""

import re


CONTEXTS = [
    "I had bread this morning and I drink tea with",
    "I had sugar this morning and I drink tea with",
]
FEATURES = ["bread", "sugar"]
INITIAL_CENTRES = [[0.2, 0.1], [0.1, 0.2]]


def encode(text):
    words = re.findall(r"[a-z]+", text.lower())
    return [float(words.count(word)) for word in FEATURES]


def distance_squared(point, centre):
    # Euclidean distance SQUARED. No square root is needed for nearest comparison.
    return sum((x - c) ** 2 for x, c in zip(point, centre))


def assign(points, centres):
    # Return a group ID for each point. Group IDs have no built-in word meaning.
    return [min(range(len(centres)), key=lambda j: distance_squared(x, centres[j]))
            for x in points]


def objective(points, centres, groups):
    # Within-group sum of squared distances, NOT prediction-versus-label error.
    return sum(distance_squared(x, centres[g]) for x, g in zip(points, groups))


def fit(points, initial_centres=None, verbose=False):
    if not points:
        raise ValueError("Need at least one point.")
    centres = [row.copy() for row in (INITIAL_CENTRES if initial_centres is None else initial_centres)]
    for iteration in range(1, 11):
        groups = assign(points, centres)
        old_objective = objective(points, centres, groups)
        updated = []
        for group_id, old_centre in enumerate(centres):
            members = [x for x, g in zip(points, groups) if g == group_id]
            # Keep an empty group's old centre rather than divide by zero.
            # Real clustering workflows may instead reinitialise empty groups.
            if not members:
                updated.append(old_centre.copy())
            else:
                updated.append([sum(x[j] for x in members) / len(members)
                                for j in range(len(old_centre))])
        new_groups = assign(points, updated)
        if verbose:
            print(f"Iteration {iteration}: centres {centres} -> {updated}")
            print("  Group IDs:", new_groups)
            print(f"  Objective {old_objective:.6f} -> {objective(points, updated, new_groups):.6f}")
        movement = max(distance_squared(a, b) for a, b in zip(centres, updated))
        centres = updated
        if movement < 1e-12:
            return centres, new_groups
    return centres, assign(points, centres)


def main():
    points = [encode(text) for text in CONTEXTS]
    print("UNSUPERVISED: input contexts only; no correct next-word labels.")
    for text, point in zip(CONTEXTS, points):
        print(text, "->", point)
    print("\nInitial squared distances to groups 0 and 1:")
    for point in points:
        print(point, "->", [round(distance_squared(point, c), 4) for c in INITIAL_CENTRES])
    centres, groups = fit(points, verbose=True)
    print("\nFINAL GROUPS (names are numeric, not next-word predictions)")
    for text, group in zip(CONTEXTS, groups):
        print(f"  Group {group}: {text}")
    print("Final centres:", centres)
    print("Two points and two clusters is a deliberately trivial demonstration.")
    print("The feature choice already exposes the bread/sugar distinction.")


if __name__ == "__main__":
    main()
