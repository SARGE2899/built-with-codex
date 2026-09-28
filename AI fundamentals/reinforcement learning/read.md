# Reinforcement learning: choose an action, then learn from reward

The defining idea here is: **the model chooses a word and receives feedback
about that chosen action**. It changes its parameters to favour higher reward.

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

## The task and its reward rule

An action means selecting bread or sugar as the completion. We explicitly
write a toy environment with this rule:

| Context | Chosen bread | Chosen sugar |
| --- | ---: | ---: |
| Earlier clue is bread | +1 | -1 |
| Earlier clue is sugar | -1 | +1 |

The environment contains our desired behaviour. It is not discovering which
completion is universally correct English. During each training step, the
policy receives the reward for the action it actually chose, rather than a
one-hot vector containing the correct answer.

This is a **contextual bandit**: one context, one action, one immediate reward.
It is a minimal reinforcement-learning setting. It does not cover delayed
rewards, long sequences of decisions, or a complete LLM preference-training
system. We deliberately chose a reward rule that resembles our labelled task
so the examples can be compared; not all RL tasks have such a simple rule.

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

## Step 1: sample a word from the probabilities

Initially probabilities are [0.5,0.5]. Training draws a random number between
0 and 1. Below 0.5 chooses bread; otherwise it chooses sugar.
The random seed is 7 to make the demonstration repeatable. The first draw is
about 0.3238, so the first chosen action is bread.

Sampling lets the model try either action. The final display instead selects
the highest-probability action. These are different selection procedures.

## Step 2: receive reward

The first context contains the earlier bread clue. The chosen action is bread,
so our environment returns reward R=+1. It returns this AFTER the choice.

## Step 3: calculate a training signal

We use a basic policy-gradient method called REINFORCE. Its objective is to
increase expected reward. For the sampled action, we use this surrogate loss:

```text
L = -R * ln(P(chosen action))
```

Here ln is the natural logarithm. With R=1 and chosen probability 0.5:

```text
L = -1 * ln(0.5) = approximately 0.693147
```

Why this expression? The logarithm increases as the probability increases.
For positive reward, reducing the negative log term encourages the chosen
action's probability to increase. Negative reward reverses the direction.
This sampled update estimates how to improve expected reward; we do not
require this one-action loss to fall monotonically across different episodes.
Negative rewards can produce negative surrogate loss values.

Differentiating through softmax gives:

```text
score_gradient[j] = R * (P(j) - indicator(j is the CHOSEN action))
```

The indicator is 1 for the chosen action, 0 for the other action. It is NOT
the supervised correct-answer vector. The sampled action and reward are held
fixed when calculating this gradient. No derivative is taken through the
environment's reward function.

For our first chosen action bread and reward +1:

```text
bread score gradient = +1*(0.5-1) = -0.5
sugar score gradient = +1*(0.5-0) = +0.5
```

## Step 4: update the parameters

Use learning rate 0.2 and input [1,0]:

```text
new parameter = old parameter - 0.2 * gradient
weight gradient = score gradient * input feature
bias gradient = score gradient
```

| Parameter | Old | Gradient | New |
| --- | ---: | ---: | ---: |
| Bread output, bread input | 0 | -0.5 | 0.1 |
| Bread output, sugar input | 0 | 0 | 0 |
| Sugar output, bread input | 0 | +0.5 | -0.1 |
| Sugar output, sugar input | 0 | 0 | 0 |
| Bread bias | 0 | -0.5 | 0.1 |
| Sugar bias | 0 | +0.5 | -0.1 |

The new scores for this input are +0.2 and -0.2:

```text
P(bread) = exp(0.2)/(exp(0.2)+exp(-0.2)) = 0.598688
P(sugar) = 0.401312
```

The rewarded action rose from 50% to about 59.87%.

## What if it had chosen sugar instead?

At the same initial probabilities, choosing sugar in the bread context earns
R=-1. The chosen-action indicator would now be [0,1]:

```text
bread gradient = -1*(0.5-0) = -0.5
sugar gradient = -1*(0.5-1) = +0.5
```

This reduces the penalised sugar action's probability. At this symmetric
starting point it happens to give the same update as rewarding bread. That
equality is a property of this particular example, not a universal RL rule.
If reward were 0, both gradients would be 0 in this no-baseline implementation.

## Repeat and inspect the results

One round visits both contexts. 100 rounds give 200 one-action episodes.
Every episode samples an action using the current probabilities and then
updates the parameters. The seed makes the following run repeatable:

| Rounds | Episodes | Cumulative mean reward | Bread probability on bread context | Sugar probability on sugar context |
| --- | ---: | ---: | ---: | ---: |
| 1 | 2 | 0.00 | 55.48% | 53.99% |
| 10 | 20 | 0.50 | 78.83% | 83.01% |
| 50 | 100 | 0.78 | 96.72% | 96.38% |
| 100 | 200 | 0.89 | 97.96% | 97.93% |

The first full round differs from the first update because it includes the
second context too. At the end there were 189 rewarded and 11 penalised choices:

```text
Mean reward = (189*(+1) + 11*(-1))/200 = 0.89
Fraction of rewarded training choices = 189/200 = 94.5%
```

94.5% describes historical sampled actions throughout training. The roughly
98% probabilities describe the final policy on these inputs. They are not
the same statistic, and neither measures performance on unseen language.
Do not conclude RL is better than supervised learning from these numbers:
the objectives and update sizes differ, despite sharing a learning-rate value.

## Follow the code

`sample_action` performs the random choice. `environment_reward` implements
our explicit feedback rule. `policy_score_gradients` calculates the sampled
policy-gradient signal. `apply_update` changes weights and biases. `train`
repeats context -> prediction -> sampled action -> reward -> update.
`show_predictions` prints probabilities and selects their maximum for display.

## Small experiments

Change `SEED` to see a different action history. Change `RATE` to 0 to confirm
that rewards alone do not change a model without parameter updates. Try fewer
rounds to observe a less trained policy. If you change the environment's reward
rule, you change the behaviour being encouraged. A poorly chosen reward can
teach unwanted behaviour even when the training algorithm is working correctly.

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
