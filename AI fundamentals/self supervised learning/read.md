# Self-supervised learning: take the answer from the text

The defining idea is: **the training data itself supplies the target**.
We remove a word from the input and ask the model to predict that word.
The model does not invent the correct answer.

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

## Exactly where the answer comes from

These are the complete sentences in `TEXT_DATA`:

```text
I had bread this morning and I drink tea with bread
I had sugar this morning and I drink tea with sugar
```

For each sentence, the code separates the final word from everything before it:

| Complete text ends with | Input passed to model | Extracted target |
| --- | --- | --- |
| tea with bread | I had bread this morning and I drink tea with | bread |
| tea with sugar | I had sugar this morning and I drink tea with | sugar |

`words[:-1]` means all words except the last. `words[-1]` means the last word.
The final answer is withheld BEFORE counting input features. Therefore the
first input has one occurrence of bread, not two. The earlier bread is a valid
input clue; the final bread is the target. Encoding the full sentence would
accidentally give the model information it would not have at prediction time.

We generate ONE training pair per sentence to keep the comparison exact.
Typical next-token language-model training uses many token positions, with
each position predicting the following token from the preceding context.
This script does not implement that larger procedure.

## The mathematical difference from supervised learning

In this deliberately matched example, there is NO difference in the numerical
training update. Supervised learning receives (input, answer) pairs directly.
Self-supervised learning constructs those pairs from complete text. Once the
pairs are constructed, the feature vectors and target vectors are identical.
The scripts use the same initialization, order, loss, and learning rate, so
their weights and predictions end up identical too.

## The tiny prediction model

There are four adjustable weights and two adjustable biases:

```text
                    input bread   input sugar
output bread            w00           w01
output sugar            w10           w11

bread score = w00*x0 + w01*x1 + b0
sugar score = w10*x0 + w11*x1 + b1
```

All six parameters start at zero. These initial numbers are our choice, not
values that have already been learned. Zero initialization works for this
simple linear classifier; it is not a recommendation to initialize every
weight in a multilayer neural network to zero.

Softmax converts the two scores to probabilities:

```text
P(bread) = exp(bread score) / (exp(bread score) + exp(sugar score))
P(sugar) = exp(sugar score) / (exp(bread score) + exp(sugar score))
```

Initially both scores are 0. Since exp(0)=1, each probability is 1/(1+1)=0.5.
The code subtracts the maximum score before exponentiating to avoid unnecessarily
large numbers; this gives exactly the same probabilities mathematically.

During prediction, the model receives only the encoded input and its parameters.
It does not receive the answer. During training, feedback is used to change the
parameters so that later predictions improve on the chosen task.

## Calculate the first training update by hand

Take the bread context, x=[1,0], with desired output bread.
Output order is [bread,sugar], so the target vector is y=[1,0].
The target 0 for sugar means "not the desired class for this example"; it does
not mean sugar is always wrong.

Our loss is mean squared error across the TWO outputs:

```text
L = ((P(bread)-1)^2 + (P(sugar)-0)^2) / 2
  = ((0.5-1)^2 + (0.5-0)^2) / 2
  = (0.25 + 0.25) / 2
  = 0.25
```

The division by 2 averages the two output errors. If we later report the mean
loss over both training examples, that is a separate averaging operation.
The loss measures the current mismatch; it does not itself tell us which weight
to change. For that we calculate gradients: how the loss changes when a number
changes slightly.

First differentiate with respect to each probability:

```text
dL/dP(bread) = 2/2 * (0.5-1) = -0.5
dL/dP(sugar) = 2/2 * (0.5-0) = +0.5
```

Both probabilities depend on both scores. Increasing the bread score increases
its probability and reduces the sugar probability. We must account for this:

```text
dP(bread)/d(bread score) = 0.5*(1-0.5) = 0.25
dP(sugar)/d(bread score) = -0.5*0.5 = -0.25

dL/d(bread score) = (-0.5)*0.25 + (+0.5)*(-0.25) = -0.25
dL/d(sugar score) = (-0.5)*(-0.25) + (+0.5)*0.25 = +0.25
```

This is the chain rule: combine how the loss depends on probabilities with how
probabilities depend on scores. The compact equivalent used in the code is:

```text
dp[j] = (2 / number_of_outputs) * (p[j] - target[j])
common = sum(p[j] * dp[j] for each output j)
score_gradient[j] = p[j] * (dp[j] - common)
```

Here common=0.5*(-0.5)+0.5*(0.5)=0.
Do not replace this with just p-target: that shortcut applies to a different
combination, softmax with cross-entropy, not our squared-error loss.

Because score=sum(weight*input)+bias:

```text
weight gradient = score gradient * corresponding input
bias gradient = score gradient
new parameter = old parameter - learning_rate * gradient
```

We choose learning_rate=0.2. With input [1,0], the first update is:

| Parameter | Old value | Gradient | New value |
| --- | ---: | ---: | ---: |
| Bread output, bread input | 0 | -0.25*1 | 0.05 |
| Bread output, sugar input | 0 | -0.25*0 | 0 |
| Sugar output, bread input | 0 | +0.25*1 | -0.05 |
| Sugar output, sugar input | 0 | +0.25*0 | 0 |
| Bread bias | 0 | -0.25 | 0.05 |
| Sugar bias | 0 | +0.25 | -0.05 |

The program does all of this automatically; a person does not manually choose
the direction of each update. The learning rate controls the size of the step.

Predict the same input again:

```text
bread score = 0.05*1 + 0*0 + 0.05 = 0.1
sugar score = -0.05*1 + 0*0 - 0.05 = -0.1
P(bread) = exp(0.1)/(exp(0.1)+exp(-0.1)) = 0.549834
P(sugar) = 0.450166
new loss = ((0.549834-1)^2 + 0.450166^2)/2 = approximately 0.202649
```

Bread probability increased from 50% to about 54.98%, and loss decreased from
0.25 to about 0.202649. These are numbers after ONE example's update.
Next the sugar example changes its active weights and the shared biases.
That is why the output printed after a complete epoch differs from this first
update. Shared parameters mean an update for one example can affect another.

## Repeat the training loop

One epoch means visiting both examples once. We use 100 epochs, giving 200
parameter updates. Every update recomputes predictions and gradients from the
current parameters. We do not reuse the initial gradients for every step.

The scripts report loss after each selected epoch using the updated model:

| Completed epochs | Mean training MSE | Bread probability on bread context | Sugar probability on sugar context |
| --- | ---: | ---: | ---: |
| 0 | 0.250000 | 50.00% | 50.00% |
| 1 | 0.225088 | 52.38% | 52.73% |
| 10 | 0.097132 | 68.19% | 69.49% |
| 50 | 0.018744 | 86.19% | 86.42% |
| 100 | 0.008503 | 90.74% | 90.82% |

These percentages are model probabilities on the training inputs, not accuracy
on unseen sentences. Choosing the highest probability selects bread for the
first input and sugar for the second. At the initial tie, the display chooses
bread because it is first in the list, not because it has learned anything.

We retain squared error because it connects to the earlier lessons. Language
models commonly use cross-entropy for next-token training. The source of a
target determines supervised versus self-supervised learning here, not the
choice between squared error and cross-entropy.

## Follow the code in order

`make_pairs(TEXT_DATA)` is the part specific to this lesson. It tokenizes each
sentence, separates its final word, and checks that the final word is one of
the two supported outputs. It raises an error for a sentence without a prefix
or a target outside the output list, instead of silently making up a mapping.

After that, `train` follows exactly the supervised procedure: encode, predict,
calculate MSE, differentiate, and update. `predict` never receives the full
training sentence or its final word.

## Small experiments

Compare the final numbers with the supervised lesson: they match. Change only
the final word of the first complete sentence from bread to sugar. The input
features stay [1,0], but its target becomes sugar. Now both examples ask for
sugar. This demonstrates that the target came from the final word, rather than
a built-in belief that the earlier bread must imply bread.

Try adding a sentence ending in coffee. The current program rejects it because
coffee is not in `OUTPUTS`. Expanding the task requires extending the output
space and corresponding parameter dimensions; merely mentioning a word in
text does not automatically create a new output neuron in this code.

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
