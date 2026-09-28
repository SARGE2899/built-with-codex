# Unsupervised learning: group inputs without next-word answers

The defining idea here is: **we supply inputs but no desired next-word labels**.
The algorithm groups their numerical representations using a distance rule.
This lesson uses K-means. It does not produce next-word probabilities.

## The same example across all four lessons

We keep the original sentence:

> I had bread this morning and I drink tea with ______

We explicitly add just ONE comparison, used in every lesson:

> I had sugar this morning and I drink tea with ______

These are the entire toy dataset. No other sentences are hidden in the program.
For the prediction lessons, the only possible outputs are `[bread, sugar]`.
We choose bread for the first example and sugar for the second as the behaviour
we want to teach. They are not the only grammatically possible completions.

To focus on the learning method, we count two words in the input:

```text
Feature order: [number of occurrences of bread, number of occurrences of sugar]
Bread context -> [1, 0]
Sugar context -> [0, 1]
```

The program lowercases the text, extracts words, and counts these two clues.
The numbers are features chosen by us. This is not a learned embedding or an
attention calculation. All other words and word order are ignored. That makes
this example small enough to calculate by hand, but limits what it can learn.
An actual language model would use a much richer representation.

## Where this lesson fits

| Type | Information supplied | What this example learns |
| --- | --- | --- |
| Supervised | Input plus a separately supplied desired word | Predict the supplied word |
| Self-supervised | Complete text; program withholds its last word | Predict that withheld word |
| Unsupervised | Input contexts without desired next words | Group similar feature vectors |
| Reinforcement | Context, chosen action, then reward | Prefer actions that earn more reward |

These are teaching categories, not four completely separate technologies.
Self-supervised learning is often placed under the wider unsupervised umbrella.
A real system can combine several of these methods.

## What is supplied and what is learned

We supply the two input contexts, the feature definition, the number of groups
(K=2), and two starting centres. We do not supply an answer saying "the next
word must be bread" or "this sentence belongs to group 0".

The algorithm learns centre locations and group assignments. The group numbers
have no intrinsic meaning: group 0 is not a vocabulary entry for bread.
The bread/sugar distinction is already exposed by our chosen features, so
this is not evidence that the algorithm discovered language meaning itself.

## Step 1: represent the inputs

```text
A = bread context = [1,0]
B = sugar context = [0,1]
```

## Step 2: choose initial centres

For reproducible arithmetic we choose:

```text
Centre 0 = [0.2,0.1]
Centre 1 = [0.1,0.2]
```

These are illustrative starting values, not answers calculated from a hidden
dataset. Centres may have fractional coordinates even when word counts are
integers, because a centre is an average of its member vectors.

## Step 3: measure squared distance to each centre

For two coordinates the squared Euclidean distance is:

```text
distance_squared = (x0-c0)^2 + (x1-c1)^2
```

The terms are ADDED. No square root is needed to compare which distance is
smaller, since the square root preserves the ordering of nonnegative values.

For input A=[1,0]:

```text
to Centre 0: (1-0.2)^2 + (0-0.1)^2 = 0.64+0.01 = 0.65
to Centre 1: (1-0.1)^2 + (0-0.2)^2 = 0.81+0.04 = 0.85
```

A joins group 0 because 0.65 < 0.85.

For input B=[0,1]:

```text
to Centre 0: (0-0.2)^2 + (1-0.1)^2 = 0.04+0.81 = 0.85
to Centre 1: (0-0.1)^2 + (1-0.2)^2 = 0.01+0.64 = 0.65
```

B joins group 1. The initial objective, the total squared distance of each
point to its assigned centre, is 0.65+0.65=1.30.

## Step 4: update the centres using averages

For each group, average each coordinate of its members. Our tiny dataset gives
one point per group:

```text
New Centre 0 = average([1,0]) = [1,0]
New Centre 1 = average([0,1]) = [0,1]
```

There is no desired next-word answer in this calculation. If a group had
multiple members, we would add their first coordinates and divide by the
number of members, then do the same for the second coordinates. No additional
members are used in this program.

Each point now sits exactly on its centre, so the objective becomes 0+0=0.
On the next iteration, assignments and centres remain unchanged and we stop.

| Iteration | Centre 0 | Centre 1 | Objective after update |
| --- | --- | --- | ---: |
| Initial | [0.2,0.1] | [0.1,0.2] | 1.30 |
| 1 | [1,0] | [0,1] | 0 |
| 2 | [1,0] | [0,1] | 0 |

Two points and two clusters make this deliberately trivial. A zero objective
here does not mean the system understands either sentence.

## Is this the same as our squared prediction error?

Both use squared differences, but compare different things:

```text
Supervised MSE: prediction versus supplied target
This K-means objective: input point versus its assigned centre
```

There is still a mathematical objective without answer labels. "Unsupervised"
does not mean "no objective, no data, and no decisions by humans".

## Where is the learning rate?

This K-means implementation does not use one. It alternates between nearest
centre assignments and exact mean updates. Not every learning algorithm uses
neural networks or gradient descent. Other unsupervised methods may use them.

## How this relates to the KNN you already know

KNN normally uses labelled reference examples: find nearby examples and use
their labels to predict a label. K-means uses no reference answer labels here:
it repeatedly finds groups and changes their centres. Both can use distance,
but distance plays a different role in their procedures.

## Follow the code

`distance_squared` calculates the formula above. `assign` finds the nearest
centre for every input. `objective` adds within-group squared distances.
`fit` repeats assignment and averaging, stopping when squared centre movement
is below 1e-12 or after ten iterations. An empty group keeps its old centre
instead of dividing by zero. A tie chooses the first centre in the list.

## Small experiments

Swap the order of the two initial centres. The numeric group IDs switch, but
the underlying separation stays the same. This shows that group IDs are
arbitrary. Try a single centre: both examples share one group, whose centre
becomes [0.5,0.5]. Its objective is 0.5+0.5=1.0.

Changing K changes the problem we ask the algorithm to solve. More clusters
can lower this training objective without producing a more useful grouping.
The program never outputs a next word, so do not read group 0 as a prediction
of bread or compare its zero objective directly with the other lessons' loss.

## Reading the Python

Open `code.py` alongside these notes. The comments explain each calculation.
`def` defines a reusable function; `return` hands its result back to the caller.
`for` repeats work. `enumerate(values)` gives both a position and its value,
starting with position 0. For example, enumerate([0.5,0.5]) visits (0,0.5) and
(1,0.5). `zip(a,b)` pairs corresponding items. `sum(...)` adds values together.
A list such as `[1,0]` stores ordered numbers; the order gives each number its
meaning. Changing feature or output order requires changing the associated
calculations consistently.

The `if __name__ == "__main__":` line runs the demonstration when the file is
executed directly. Importing it for inspection does not automatically train.

## How to use these files

This `read.md` is a reading document, not a program. Read it in GitHub or a
Markdown preview. To optionally reproduce the calculations, open a terminal
in this lesson's folder and run:

```sh
python code.py
```

Use Python 3. No additional packages, API keys, internet access, GPU, or model
download are needed. The program prints results and exits. It does not save
weights, so running it again starts training from the beginning.

## What this demonstration does not establish

Two carefully chosen contexts cannot establish general language understanding.
The feature extractor ignores most of the sentence. A new sentence with the
same bread/sugar counts has the same representation, even if its meaning is
different. There is no held-out test dataset here. Training results show that
this tiny procedure works on its supplied examples; they do not prove useful
performance on arbitrary text.
