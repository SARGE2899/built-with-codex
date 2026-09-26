# Changing weights to get the desired output

This lesson follows one small model from its initial, incorrect predictions to its learned predictions. Read it alongside [code.py](code.py). Every number needed for the worked example is included here; you can read the lesson without running anything.

We use exactly two training examples:

| Input | Correct next word |
|---|---|
| I drink | tea |
| I eat | rice |

No other sentences are used. The purpose is to understand how changing weights can make the **same model give different answers for different inputs**.

## 1. What we supply, and what the model learns

We supply:

- Input words the program recognises: `i`, `drink`, `eat`.
- Possible output words: `tea`, `rice`.
- The two labelled training examples above.
- A mathematical structure: weighted sums followed by softmax.
- Starting weights, biases, learning rate, and number of training passes.

Training changes the **weights and biases**. It does not change the word lists, rewrite the equations, or invent new training sentences.

The model does not initially know that drinks relate to tea or eating relates to rice. It only receives numbers, makes predictions, measures their errors, and adjusts its parameters.

This is a linear classifier with softmax. There is no hidden layer. It is a teaching example, not a full LLM.

## 2. Four things that must not be confused

| Name | Example | Meaning |
|---|---|---|
| Input vector | `[1, 1, 0]` | Counts of input words |
| Weight row for tea | `[1, 0, 2]` | Multipliers used to calculate tea's score |
| Target vector | `[1, 0]` | Tea is the correct answer for this example |
| Predicted probabilities | `[0.2689, 0.7311]` | Current probabilities of tea and rice |

In particular, **`[1, 0, 2]` is tea's starting weight row in this lesson**. It is not the input and not the predicted output.

The order of each list matters. We consistently use:

```text
Input positions:  [i, drink, eat]
Output positions: [tea, rice]
```

The input vocabulary and output vocabulary are separate in this simplified program. We are not allowing it to predict every input word as an output.

## 3. Encoding the inputs

Each input becomes a list of word counts:

```text
I drink -> [1, 1, 0]
I eat   -> [1, 0, 1]
```

For “I drink”:

- `i` occurs once.
- `drink` occurs once.
- `eat` occurs zero times.

For “I eat”, the active words are instead `i` and `eat`.

The function `encode()` lowercases the text before splitting it at whitespace, so `I` and `i` are treated alike. It counts the resulting words in the specified positions.

These counts are not embeddings. They also discard order: “drink I” has the same representation as “I drink”. Unknown input words raise an error, and punctuation is not automatically removed.

## 4. Why the target for rice is zero in the first example

For “I drink”, our supplied correct answer is tea:

| Output choice | Correct for this example? | Target |
|---|---|---:|
| tea | Yes | 1 |
| rice | No | 0 |

So the target vector is `[1, 0]`.

For “I eat”, the correct answer is rice, so the target vector is `[0, 1]`.

The zero means **not the correct output for this particular labelled example**. It is not a word ID, a weight, or a claim that rice can never follow any other text.

Such a target is called a **one-hot vector**: one position is 1, and all others are 0.

## 5. Starting weights and biases

We deliberately start from these values:

| Candidate output | Weight for i | Weight for drink | Weight for eat | Bias |
|---|---:|---:|---:|---:|
| tea | 1 | 0 | 2 | 0 |
| rice | 1 | 1 | 2 | 0 |

There are six weights and two biases: **eight adjustable numbers**.

Each output has its own equation:

```text
tea score  = x_i*w_tea,i  + x_drink*w_tea,drink  + x_eat*w_tea,eat  + b_tea
rice score = x_i*w_rice,i + x_drink*w_rice,drink + x_eat*w_rice,eat + b_rice
```

A bias is an adjustable addition to the score. It does not get multiplied by an input count; equivalently, you can imagine it has a constant input of 1.

The code indexes weights as `weights[output_id][input_id]`. For example, `weights[0][1]` is the weight from input “drink” to output candidate “tea”. Its initial value is zero.

## 6. First forward calculation: “I drink”

Use input `[1, 1, 0]`.

Tea's score:

```text
z_tea = 1*1 + 1*0 + 0*2 + 0 = 1
```

Rice's score:

```text
z_rice = 1*1 + 1*1 + 0*2 + 0 = 2
```

The “eat” weights contribute nothing because their input count is zero.

These scores are not probabilities. Scores may be negative or larger than 1.

## 7. Softmax: scores become probabilities

For two output choices:

```text
P(tea)  = exp(z_tea)  / (exp(z_tea) + exp(z_rice))
P(rice) = exp(z_rice) / (exp(z_tea) + exp(z_rice))
```

`exp(z)` means e raised to power z, where e is approximately 2.71828.

Substitute our scores:

```text
P(tea)  = exp(1) / (exp(1) + exp(2)) = 0.268941... = 26.89%
P(rice) = exp(2) / (exp(1) + exp(2)) = 0.731059... = 73.11%
```

They add to 1, or 100%.

The initial model favours rice, even though our training target is tea. This gives training a mistake to correct.

In Python, the implementation subtracts the largest score before exponentiating. This avoids excessively large exponentials and leaves the probabilities unchanged, because the common factor cancels in the division.

For “I eat” with the initial parameters, both scores are 3. The probabilities are therefore 50% each. The program selects the first candidate, tea, in an exact tie. That tie-breaking rule is not a learned preference.

## 8. Measuring the mistake with mean squared error

We use mean squared error, abbreviated MSE, across the output choices:

```text
MSE = ((P(tea) - target_tea)^2 + (P(rice) - target_rice)^2) / 2
```

For “I drink” -> tea:

```text
MSE = ((0.268941 - 1)^2 + (0.731059 - 0)^2) / 2
    = approximately 0.534447
```

Both squared errors are equal here, because the two probabilities sum to 1 and the targets are `[1, 0]`.

### Why divide by two?

Because we are taking the mean over **two output choices**. It has nothing to do with the fact that our dataset also happens to have two examples.

With three output choices, this definition of MSE would divide the summed squared errors by three.

Sometimes people use half the sum of squared errors to simplify derivatives. For exactly two outputs, that happens to equal our MSE. For other output counts, they differ by a scaling factor.

### What does 0.534447 mean?

It is an error score. It is not 53.44% accuracy, a word count, or an amount to subtract directly from a weight.

For a perfect prediction `[1, 0]` against target `[1, 0]`, the MSE would be zero. Finite softmax scores approach these extreme probabilities rather than reaching exact mathematical 0 and 1.

## 9. Loss and gradient have different jobs

**Loss measures the mistake. A gradient tells us how a small parameter change affects that loss.**

The dependency is:

```text
weight -> score -> output probabilities -> MSE
```

To adjust a weight correctly, we follow the dependency backwards using derivatives. This is the **chain rule**.

For example, increasing the “drink -> tea” weight raises tea's score when “drink” is present. Softmax then raises tea's probability and lowers rice's probability. For the current target, both effects reduce the loss.

We calculate all gradients from the same pre-update state before changing any parameters.

## 10. First derivative: how loss depends on a probability

Let:

```text
p = P(tea)  = 0.268941...
q = P(rice) = 0.731059...
```

Our loss is:

```text
L = ((p - 1)^2 + (q - 0)^2) / 2
```

The derivative of a squared difference contributes a factor of 2, which cancels the divisor of 2:

```text
dL/dp = p - 1 = -0.731059...
dL/dq = q - 0 = +0.731059...
```

Read `dL/dp` as “how loss changes for a small change in p”. You do not need to memorise the notation to follow the numerical update.

For a general count K of outputs, the probability derivative of MSE is:

```text
dL/dp_k = (2/K) * (p_k - target_k)
```

The code keeps this general factor explicitly.

## 11. Second derivative: how softmax probabilities depend on scores

Changing tea's score affects **both** probabilities. For two outputs:

```text
dp/dz_tea = p*(1-p) = p*q
dq/dz_tea = -p*q

dp/dz_rice = -p*q
dq/dz_rice = q*(1-q) = p*q
```

For our starting probabilities:

```text
p*q = approximately 0.196612
```

The positive sign says that increasing a candidate's own score increases its probability. The negative sign says it reduces the other candidate's probability.

### Combine both routes to the loss

Tea's score gradient must include both affected probabilities:

```text
dL/dz_tea = (dL/dp)*(dp/dz_tea) + (dL/dq)*(dq/dz_tea)
          = (p-1)*(p*q) + q*(-p*q)
          = ((p-1) - q)*p*q
          = approximately -0.28746968
```

Similarly:

```text
dL/dz_rice = (p-1)*(-p*q) + q*(p*q)
           = approximately +0.28746968
```

Increasing tea's score therefore reduces the loss locally. Increasing rice's score increases it locally.

### Why not just use probability minus target?

For **softmax with cross-entropy**, the score derivative simplifies to probability minus target. That was the earlier lesson.

For **softmax with MSE**, probability minus target is only the probability derivative in this two-output case. We still need the softmax derivatives to reach the scores.

Simply reusing the earlier score-gradient shortcut would not implement gradient descent on the stated squared-error loss.

MSE is used here because we are studying it. Cross-entropy is generally the conventional objective for categorical next-token prediction. Changing the loss changes the training calculation, even when the forward predictions still use softmax.

## 12. From score gradients to weight gradients

A candidate's score is a weighted sum plus a bias:

```text
score = x1*w1 + x2*w2 + x3*w3 + bias
```

Therefore:

```text
weight_gradient = score_gradient * input_value
bias_gradient = score_gradient
```

For the current input `[1, 1, 0]`:

| Candidate | Gradient for i weight | Gradient for drink weight | Gradient for eat weight | Bias gradient |
|---|---:|---:|---:|---:|
| tea | -0.287470 | -0.287470 | 0 | -0.287470 |
| rice | +0.287470 | +0.287470 | 0 | +0.287470 |

The “eat” weight gradients are zero because they are multiplied by an input of zero. This is why inactive input features do not receive direct weight updates in this model.

## 13. Apply the first update

Use a learning rate of `0.1`:

```text
new_weight = old_weight - 0.1 * weight_gradient
new_bias = old_bias - 0.1 * bias_gradient
```

The learning rate is an adjustment multiplier, not a probability or a target loss.

### Tea's drink weight

It starts at zero:

```text
0 - 0.1*(-0.28746968) = 0.028746968
```

Subtracting a negative number increases the weight.

### Rice's drink weight

It starts at one:

```text
1 - 0.1*(+0.28746968) = 0.971253032
```

### All parameters after the update

| Candidate | i weight | drink weight | eat weight | Bias |
|---|---:|---:|---:|---:|
| tea | 1.028747 | 0.028747 | 2.000000 | +0.028747 |
| rice | 0.971253 | 0.971253 | 2.000000 | -0.028747 |

The code calculates with full floating-point precision. Tables round only for readability.

## 14. Check whether that update helped

Predict again for “I drink”, using the new parameters:

```text
tea score = 1*1.028747 + 1*0.028747 + 0*2 + 0.028747
          = approximately 1.086241

rice score = 1*0.971253 + 1*0.971253 + 0*2 - 0.028747
           = approximately 1.913759
```

Apply softmax again:

```text
P(tea)  = approximately 30.42%
P(rice) = approximately 69.58%
MSE     = approximately 0.484179
```

| Measurement | Before | After one update |
|---|---:|---:|
| Tea probability | 26.89% | 30.42% |
| Rice probability | 73.11% | 69.58% |
| MSE | 0.534447 | 0.484179 |

The prediction has improved, but rice still has the higher probability. One update need not make the answer correct.

## 15. Now train on the second example

The next example is:

```text
Input:  I eat -> [1, 0, 1]
Target: rice  -> [0, 1]
```

We use the **already updated parameters**, not the starting parameters again.

After the first update, the scores for this second input are approximately:

```text
tea score  = 1.028747 + 2 + 0.028747 = 3.057494
rice score = 0.971253 + 2 - 0.028747 = 2.942506
```

Tea now has about 52.87%, and rice has about 47.13%. The first update helped the first example but made this second example worse, because both examples share “i” and the biases.

For this second example, the target is rice. The next update therefore pushes in the other direction:

- The “eat -> rice” weight increases.
- The “eat -> tea” weight decreases.
- The shared “i” weights and biases also change.
- The “drink” weights stay unchanged during this update because the input count for drink is zero.

This explains why we must revisit **both** examples. Each update follows the current example's target; it is not guaranteed to improve every other example immediately.

After both first-pass updates, the program reports:

| Input | P(tea) | P(rice) | Correct answer |
|---|---:|---:|---|
| I drink | 28.23% | 71.77% | tea |
| I eat | 48.92% | 51.08% | rice |

The first example's tea probability fell a little from its immediately-after-first-update value of 30.42%. That is expected here: we just adjusted shared parameters for the second example.

## 16. Steps, epochs, and two different averages

One **update** in this script processes one labelled example.

One **epoch** means one complete pass through the dataset:

```text
Update on I drink -> tea
Update on I eat   -> rice
End of one epoch
```

With 200 epochs and two examples, the main model receives **400 updates**.

The order is fixed, and we update after each example. This is sequential gradient descent, often described as stochastic gradient descent with batch size one. We are not averaging both examples' gradients before each update.

There are also two different averages in the reporting:

1. Each example's MSE averages the squared errors over the **two output choices**.
2. The reported mean dataset loss averages the two example losses over the **two examples**.

Initially:

```text
MSE for I drink = 0.534447
MSE for I eat   = 0.250000

Mean dataset loss = (0.534447 + 0.250000) / 2 = approximately 0.392223
```

The two divisors happen to both be two. They count different things.

## 17. Results from the actual program

The following values come from the specified starting weights, learning rate 0.1, fixed example order, and 200 epochs:

| Completed epochs | P(tea given I drink) | P(rice given I eat) | Mean dataset MSE |
|---:|---:|---:|---:|
| 0 | 26.89% | 50.00% | 0.392223 |
| 1 | 28.23% | 51.08% | 0.377193 |
| 2 | 29.69% | 52.05% | 0.362109 |
| 5 | 34.73% | 54.45% | 0.316782 |
| 10 | 44.25% | 57.52% | 0.245612 |
| 50 | 76.98% | 77.72% | 0.051312 |
| 100 | 85.32% | 85.55% | 0.021209 |
| 200 | 90.45% | 90.52% | 0.009050 |

At the end, all possible output probabilities are:

| Input | P(tea) | P(rice) | Selected output |
|---|---:|---:|---|
| I drink | 90.45% | 9.55% | tea |
| I eat | 9.48% | 90.52% | rice |

Unlike the earlier single-example lesson, the model has learned to select different outputs for the two different inputs.

These are training-set results. We have not tested meaningful generalisation to new sentences, and the percentages are not measured real-world accuracy.

## 18. Inspect the final weights

After training:

| Candidate | i weight | drink weight | eat weight | Bias |
|---|---:|---:|---:|---:|
| tea | 1.099174 | 1.425869 | 0.673305 | +0.099174 |
| rice | 0.900826 | -0.425869 | 3.326695 | -0.099174 |

For the active “drink” input, the tea weight is much larger than the rice weight. For the active “eat” input, the rice weight is much larger than the tea weight.

The answer is determined by the **whole weighted sum and competing output score**, not by a single weight in isolation.

A negative weight is allowed. It reduces a candidate's score when its input is positive. Probabilities cannot be negative, but weights and scores can.

The shared “i” feature alone cannot distinguish the examples because it occurs in both. The distinguishing evidence is “drink” versus “eat”.

## 19. How the code is organised

| Function | What it does | Changes weights? |
|---|---|---|
| `encode()` | Turns known input words into counts | No |
| `make_target()` | Turns the correct word into a one-hot target | No |
| `forward()` | Calculates scores and softmax probabilities | No |
| `mse()` | Calculates the average squared error | No |
| `gradients()` | Calculates derivatives through MSE and softmax | No |
| `train_one()` | Uses one example to update parameters | Yes |
| `show_parameters()` | Prints weights and biases | No |
| `report()` | Prints predictions and losses for both examples | No |
| `main()` | Sets up the model, demonstrates an update, then trains | Calls the training functions |

The worked single-update demonstration uses a separate copy. It does not give the main model an extra update. The main model begins its 200 epochs from the original parameters.

## 20. Understanding the general gradient loop in Python

The code does not hardcode the special two-output expression `((p-1)-q)*p*q`. It implements the general chain rule for each score.

For output score j, it visits every probability k:

```python
indicator = 1.0 if k == j else 0.0
softmax_derivative = probabilities[k] * (indicator - probabilities[j])
score_gradient += probability_gradients[k] * softmax_derivative
```

`indicator` is 1 when k and j identify the same output, otherwise zero.

For tea's score:

- The tea-probability route uses `p*(1-p)`.
- The rice-probability route uses `q*(0-p)`.
- Both contributions are added, because both probabilities affect the loss.

Then each score gradient is multiplied by each input value to get the corresponding weight gradients.

This is the precise mathematical work hidden behind the phrase “the model adjusts its weights”.

## 21. Python syntax guide

### Lists, rows, and indices

Python list positions start at zero. `OUTPUT_WORDS[0]` is tea and `OUTPUT_WORDS[1]` is rice.

`weights[0]` selects tea's complete weight row. `weights[0][1]` selects the drink weight within that row.

### Tuples and unpacking

Each training example is a pair, such as `("I drink", "tea")`.

```python
for text, correct_word in TRAINING_DATA:
```

Unpacks each pair into its two named variables.

### List comprehensions

```python
[1.0 if word == correct_word else 0.0 for word in OUTPUT_WORDS]
```

Builds a new list by examining each possible output. For target tea, it creates `[1.0, 0.0]`.

### `.copy()` and separate model states

```python
weights = [row.copy() for row in INITIAL_WEIGHTS]
```

Copies each inner list. Copying only the outer list would leave the nested rows shared and could accidentally change the initial values during training.

### `zip()`

`zip(probabilities, target)` pairs each predicted probability with the target for the same output. Both lists must use the same output ordering.

### `** 2`, `*`, and `-=`

`difference ** 2` squares a difference. `a * b` multiplies two values.

`weight -= adjustment` means `weight = weight - adjustment`.

### `range()`

`range(1, EPOCHS + 1)` produces epoch numbers 1 through EPOCHS. The upper endpoint is excluded.

### `max(..., key=...)`

The final selection compares probabilities and returns the index of the largest one. Its `lambda` is a short function specifying the comparison value. No random sampling is used.

### Formatting

`{probability:.2%}` displays a proportion as a percentage with two decimal places. `{loss:.6f}` displays six decimal places. Display rounding does not change the numbers used for training.

### Returning multiple values

`forward()` returns `scores, probabilities`. A caller can unpack both, or use `_` for an unneeded result:

```python
_, probabilities = forward(inputs, weights, biases)
```

Here `_` is an ordinary variable name conventionally meaning “this returned value is not needed”.

### The main guard

`if __name__ == "__main__":` calls `main()` when the script is executed directly, rather than automatically running the lesson when another script imports it.

The model parameters are created inside `main()`, so each call starts a fresh model. No trained parameters are saved to disk.

## 22. Reading the results without running the program

Follow sections 6 through 14 first. They show one complete update for one example.

Then read sections 15 through 18 to see why the second example changes different weights and how repeated training reaches two different answers.

The script is optional: if you execute `python code.py` in a folder containing it, it prints these stages using only Python's standard library. It finishes automatically and does not ask for interactive input.

No installation instructions or machine-specific paths are needed to understand this document.

## 23. Optional experiments with the same two examples

### Set the learning rate to zero

With `LEARNING_RATE = 0.0`, no parameters change. All predictions remain at their initial values. This isolates the role of the update multiplier.

### Use only one epoch

With `EPOCHS = 1`, both examples receive one update. Compare that result with the separate one-example demonstration. They are not the same point in training.

### Reduce the learning rate

Try `0.01` instead of `0.1`. The changes are smaller, so this example generally needs more passes to reach comparable probabilities. Arbitrarily increasing the rate is not guaranteed to help; excessively large steps can be unstable.

### Reverse the example order

Swap the order of the two pairs in `TRAINING_DATA`. Because each update uses the parameters left by the preceding update, the trajectory can change. That does not mean the formulas are inconsistent.

### Exchange the targets deliberately

If you intentionally relabel the examples, the model will follow those labels. It has no independent knowledge that your labels are reasonable. Restore the original examples afterwards.

### Try “drink I”

It encodes identically to “I drink”, demonstrating the loss of word order. The model cannot distinguish information its representation discards.

## 24. What “desired output” means here

It means the output specified by the labelled training example. We use that target to guide parameter updates.

Training does not guarantee any arbitrary set of desired outputs can be achieved by any model. Conflicting targets, insufficient model capacity, unsuitable inputs, and optimisation issues can prevent that.

For this tiny dataset, the input features distinguish the two examples sufficiently for this model to learn both choices.

We still should not conclude that the model understands language. It cannot produce “with sugar”, because neither word is an output choice. It has no instruction for continuing a sentence after the first prediction, and no evidence of performance on new sentences.

## 25. Check your understanding

1. Is `[1, 0, 2]` an output probability list in this lesson?
2. Why do we subtract zero from rice's prediction when training “I drink”?
3. Why does MSE divide by two?
4. Why can't we subtract the loss directly from every weight?
5. Why do the eat weights stay unchanged during the first update?
6. Why can the next example partly undo an improvement to the first one?
7. How many updates occur in 200 epochs?

Answers:

1. No. It is tea's initial weight row, ordered by input features `[i, drink, eat]`.
2. Rice is not the correct class for that example, so its target is zero.
3. We average over two output choices.
4. The loss alone gives no parameter-specific direction; derivatives provide that information.
5. The input value for eat is zero, making those weight gradients zero.
6. Both examples use the shared i weights and biases, so their updates interact.
7. The main model receives 400: two examples per epoch times 200 epochs. The printed single-update demonstration uses a separate copy.

The complete loop is **input counts -> scores -> probabilities -> squared error -> gradients -> updated weights**. Repeating it across both labelled examples teaches the model to favour tea for “I drink” and rice for “I eat”.
