# Predicting the next word using existing words

This guide explains the accompanying `code.py` using the same small example throughout. Read it alongside the code. You do not need to know transformers, embeddings, or calculus to follow the worked calculation.

Our question is:

> Given **“I drink”**, can a small mathematical model learn to predict **“tea”**?

We will not quietly introduce extra sentences. Every training update in this program uses the same example.

## 1. Exactly what we give the program

There are three different things:

| Thing | Actual value | Purpose |
|---|---|---|
| Vocabulary | `['i', 'drink', 'tea', 'coffee']` | Lists the words the program recognises and can predict |
| Training sentence | `'i drink tea'` | Provides an example with a known answer |
| Input after training | `'I drink'` | The text we want to complete |

The vocabulary is **not itself a set of training sentences**. Its list order assigns IDs; it does not teach that coffee should come after tea.

Knowing that tea and coffee exist is different from being shown which one follows “i drink” in an example.

We create exactly one training pair:

```text
Input:          i drink
Correct answer: tea
```

Coffee is an allowed candidate. It is never the correct answer in this program's training data.

### Why do we already know the answer?

During training, knowing the answer lets us measure mistakes. We hide the final word from the prediction calculation, predict it, and then use the correct answer to calculate an update.

This is similar to practising a question and checking the answer afterwards.

During the final prediction, the prediction function receives the input numbers and uses the learned weights and biases. It does not receive the correct answer as an argument.

However, we deliberately test on the same input used for training. This demonstrates learning one example, not success on unseen data.

## 2. How to read this guide

This is a reading guide. You can follow every calculation without running the program or installing anything.

Keep the accompanying [code.py](code.py) open alongside this document to connect each explanation to its Python implementation. GitHub displays this Markdown document with formatted headings, tables, and code examples.

Read the sections in order:

1. Start with the vocabulary, training pair, and input counts.
2. Follow how the weights produce scores and softmax turns them into probabilities.
3. Work through the first weight update using the actual numbers.
4. Compare the before-and-after predictions, then review the Python syntax and optional experiments.

The input stays `"I drink"`, and the training answer stays `"tea"` throughout the main walkthrough. Expected results are included beside the calculations so you can check your understanding as you read.

## 3. What kind of model is this?

It is a **linear classifier with softmax**, trained to choose one of four words. You may also hear this called multinomial logistic regression.

Its calculation is:

```text
Input word counts
    -> four weighted sums
    -> softmax probabilities
    -> choose the highest-probability word
```

Compared with your Excel neural-network exercise:

| Your Excel exercise | This program |
|---|---|
| Numerical inputs | Four word-count inputs |
| Adjustable weights and biases | Adjustable weights and biases |
| Hidden neurons | No hidden layer in this simplified model |
| Sigmoid calculations | Softmax across the four output scores |
| Weight-update row | Weight-update loop repeated 200 times |

This is not a full LLM. It is intentionally small enough that we can calculate its first training update by hand.

## 4. Assigning IDs: `enumerate()`

The vocabulary is:

```python
VOCABULARY = ["i", "drink", "tea", "coffee"]
```

Python counts list positions starting at zero:

| Position / ID | Word |
|---|---|
| 0 | i |
| 1 | drink |
| 2 | tea |
| 3 | coffee |

`enumerate()` gives both the position and the item:

```python
for index, word in enumerate(VOCABULARY):
    print(index, word)
```

Output:

```text
0 i
1 drink
2 tea
3 coffee
```

Our code uses a dictionary comprehension:

```python
WORD_TO_ID = {word: index for index, word in enumerate(VOCABULARY)}
```

The longer equivalent is:

```python
WORD_TO_ID = {}
for index, word in enumerate(VOCABULARY):
    WORD_TO_ID[word] = index
```

Both create:

```python
{"i": 0, "drink": 1, "tea": 2, "coffee": 3}
```

The IDs are labels. Coffee having ID 3 does not mean it is larger, better, or more meaningful than tea.

## 5. Preparing the training pair

These lines split the sentence:

```python
TRAINING_SENTENCE = "i drink tea"
training_words = TRAINING_SENTENCE.split()
```

`split()` separates the text at whitespace:

```python
training_words == ["i", "drink", "tea"]
```

Then:

```python
training_input = " ".join(training_words[:-1])
correct_next_word = training_words[-1]
correct_word_id = WORD_TO_ID[correct_next_word]
```

Breakdown:

| Expression | Meaning | Result |
|---|---|---|
| `training_words[:-1]` | Everything before the last item | `['i', 'drink']` |
| `' '.join(...)` | Join those items with spaces | `'i drink'` |
| `training_words[-1]` | The last item | `'tea'` |
| `WORD_TO_ID['tea']` | Look up tea's ID | `2` |

This script only uses **“i drink” -> “tea”**. It does not separately train **“i” -> “drink”**. That would be another training pair, and would require changing the training loop.

## 6. Turning “I drink” into four numbers

Our `encode()` function counts words. The positions always mean:

```text
[count of i, count of drink, count of tea, count of coffee]
```

For “I drink”:

1. Start with `[0, 0, 0, 0]`.
2. Lowercase the input: “i drink”.
3. Find “i”: increase position 0 -> `[1, 0, 0, 0]`.
4. Find “drink”: increase position 1 -> `[1, 1, 0, 0]`.

So:

```text
I drink -> [1, 1, 0, 0]
```

The code uses floating-point numbers, so you see `[1.0, 1.0, 0.0, 0.0]`. They represent the same counts.

### A distinction worth keeping clear

| Representation | Example | What it means |
|---|---|---|
| Word ID | `tea -> 2` | Which word we mean |
| Input counts | `I drink -> [1, 1, 0, 0]` | Which words occur in this input |
| Target vector | `tea -> [0, 0, 1, 0]` | Which output should be correct |

These are three different jobs. Our input counts are **not learned embeddings**.

### What if the input contains an unknown word?

The function raises `ValueError`. For example, “I drink water” fails because “water” is outside this vocabulary.

The tokenizer is intentionally basic: it lowercases and splits at whitespace. It does not remove punctuation, so `tea.` and `tea` are different strings here.

### What happens to word order?

This representation loses order:

```text
I drink -> [1, 1, 0, 0]
drink I -> [1, 1, 0, 0]
```

That is a limitation of this teaching model. Real LLMs account for token positions and context.

## 7. The weights: one equation for each candidate word

We have four inputs and four candidate outputs. Each output has four weights and one bias.

That means **16 weights + 4 biases = 20 adjustable numbers**.

The code stores the weights as a table:

| Output candidate | Weight for input i | Weight for input drink | Weight for input tea | Weight for input coffee | Bias |
|---|---:|---:|---:|---:|---:|
| i | 0 | 0 | 0 | 0 | 0 |
| drink | 0 | 0 | 0 | 0 | 0 |
| tea | 0 | 0 | 0 | 0 | 0 |
| coffee | 0 | 0 | 0 | 0 | 0 |

All start at zero in this example.

`weights[output_id][input_id]` selects one number from that table. For example, `weights[2][1]` is the weight connecting input “drink” to output candidate “tea”.

For **every** candidate word, the equation is:

```text
score = x1*w1 + x2*w2 + x3*w3 + x4*w4 + bias
```

Each candidate uses its own weights and bias. The input is the same `[1, 1, 0, 0]` for all four equations.

Before training, the tea score is:

```text
tea score = 1*0 + 1*0 + 0*0 + 0*0 + 0 = 0
```

Every other score is also zero.

Zero initialization works for this simple classifier. Do not assume that initializing every weight to zero is appropriate for a network with hidden layers.

## 8. Softmax converts scores to probabilities

A score is not yet a probability. It can be negative or larger than 1.

For tea, softmax computes:

```text
P(tea) = exp(tea score)
         -----------------------------------------------------------
         exp(i score) + exp(drink score) + exp(tea score) + exp(coffee score)
```

`exp(z)` means e raised to the power z, where e is approximately 2.71828. In Python this is `math.exp(z)`.

At the start, all four scores are zero. Since `exp(0) = 1`:

```text
P(tea) = 1 / (1 + 1 + 1 + 1) = 0.25 = 25%
```

Each word gets 25%. All four probabilities add to 100%.

This does not mean the model has discovered that all four words are equally sensible. It means we gave it equal starting scores.

### Why softmax rather than four separate sigmoids?

Here we want one distribution over mutually exclusive next-word choices. Softmax makes the candidate probabilities add to 1.

Applying sigmoid independently to four scores would not generally make their outputs add to 1. Sigmoid still has useful roles in other model designs, including binary classification.

### Why subtract the largest score in the code?

The code calculates:

```python
largest_score = max(scores)
exponentials = [math.exp(score - largest_score) for score in scores]
```

This is numerically safer when scores become large. It gives the same probabilities because every exponential is multiplied by the same factor, which cancels between numerator and denominator.

At the start all scores are zero, so the subtraction changes nothing.

## 9. Checking the prediction: the loss

Our correct answer is tea. We measure the mistake using:

```text
loss = -ln(P(tea))
```

This is cross-entropy for one example with one correct class. `ln` means the natural logarithm; Python's `math.log()` uses this by default.

You can follow the training example without deriving logarithms. Look at what this rule does:

| Probability assigned to tea | Loss, approximately |
|---:|---:|
| 0.25 | 1.3863 |
| 0.50 | 0.6931 |
| 0.90 | 0.1054 |
| 0.99 | 0.0101 |
| 1.00 | 0 |

As the model assigns more probability to the correct answer, the loss decreases.

This is the same **role** squared error played in our earlier numerical example: a score for how wrong the prediction is. It is a different formula suited to this probability prediction task.

The value 1.3863 is a penalty score, not a number of incorrect words or a percentage accuracy.

The code uses:

```python
loss = -math.log(max(probabilities[correct_word_id], 1e-15))
```

`1e-15` is 0.000000000000001. The `max` prevents taking the logarithm of zero due to numerical rounding.

## 10. How the program knows which way to adjust

Loss measures the mistake. A **gradient** tells us how a small change to a number affects the loss.

For the particular combination of softmax and cross-entropy, the gradient with respect to each output score simplifies to:

```text
score_gradient = predicted_probability - target
```

This is a mathematical derivative, not a guessed rule and not the loss itself. This simplified formula is specific to this setup; it is not the update rule for every possible model.

The target is 1 for tea and 0 for each other candidate:

| Candidate | Starting probability | Target | Score gradient |
|---|---:|---:|---:|
| i | 0.25 | 0 | +0.25 |
| drink | 0.25 | 0 | +0.25 |
| tea | 0.25 | 1 | -0.75 |
| coffee | 0.25 | 0 | +0.25 |

The training update subtracts a small multiple of the gradient.

For tea, subtracting a negative amount increases its score-producing numbers. For the other candidates, subtracting a positive amount decreases them.

If you later study derivatives, this follows by writing the loss as `-correct_score + ln(sum(exp(all_scores)))` and differentiating with respect to each score. You do not need that derivation to work through the next section.

## 11. The complete first weight update, using actual numbers

The learning rate in the code is:

```python
learning_rate = 0.1
```

This controls how large each adjustment is. It is not a probability or a target error.

The update equations are:

```text
weight_gradient = score_gradient * input_value
new_weight = old_weight - learning_rate * weight_gradient
new_bias = old_bias - learning_rate * score_gradient
```

### Update tea's weights

Tea's score gradient is `-0.75`.

For the “i” input, the input value is 1:

```text
weight_gradient = -0.75 * 1 = -0.75
new_weight = 0 - 0.1*(-0.75) = 0.075
```

The “drink” input is also 1, so its weight also becomes `0.075`.

The “tea” and “coffee” input counts are zero:

```text
weight_gradient = -0.75 * 0 = 0
new_weight = 0 - 0.1*0 = 0
```

Their weights stay at zero. Remember: tea is the **output target**, but it is absent from the **input “i drink”**.

The tea bias becomes:

```text
new_bias = 0 - 0.1*(-0.75) = 0.075
```

### Update each other candidate

Each other candidate has score gradient `+0.25`.

For each active input:

```text
new_weight = 0 - 0.1*(0.25*1) = -0.025
```

The bias also becomes `-0.025`. Weights for the zero-valued input slots stay at zero.

### The full table after one update

| Output candidate | Weight for i | Weight for drink | Weight for tea | Weight for coffee | Bias |
|---|---:|---:|---:|---:|---:|
| i | -0.025 | -0.025 | 0 | 0 | -0.025 |
| drink | -0.025 | -0.025 | 0 | 0 | -0.025 |
| tea | +0.075 | +0.075 | 0 | 0 | +0.075 |
| coffee | -0.025 | -0.025 | 0 | 0 | -0.025 |

This is training: the program has changed its adjustable numbers using the example's error.

## 12. Predict again after that update

Our input remains `[1, 1, 0, 0]`.

The new tea score is:

```text
1*0.075 + 1*0.075 + 0*0 + 0*0 + 0.075 = 0.225
```

Each other candidate has score:

```text
1*(-0.025) + 1*(-0.025) + 0*0 + 0*0 + (-0.025) = -0.075
```

Apply softmax:

```text
P(tea) = exp(0.225) / (exp(0.225) + 3*exp(-0.075))
       = approximately 0.3103
       = approximately 31.03%
```

Each other candidate gets approximately 22.99%.

With the stable calculation in the code, the same result is:

```text
P(tea) = 1 / (1 + 3*exp(-0.3))
```

The new loss is approximately:

```text
-ln(0.3103) = 1.1701
```

Comparison:

| Quantity | Before the update | After one update |
|---|---:|---:|
| P(tea) | 25.00% | 31.03% |
| Loss | 1.3863 | 1.1701 |

The correct answer gained probability and the loss decreased. That is evidence that this update improved this example's prediction.

## 13. Repeat the same loop 200 times

The loop is:

```text
Predict probabilities
    -> measure loss
    -> calculate gradients
    -> update weights and biases
    -> repeat with the same example
```

We calculate new gradients each time because the probabilities have changed. We do not add the same `0.075` to tea's weights on every step.

The code uses:

```python
for step in range(1, training_steps + 1):
```

`range(1, 201)` produces integers from 1 through 200. The ending boundary is excluded.

Selected output from the program:

| Printed step | Updates already completed | P(tea) | Loss |
|---:|---:|---:|---:|
| 1 | 0 | 25.00% | 1.3863 |
| 2 | 1 | 31.03% | 1.1701 |
| 5 | 4 | 48.88% | 0.7157 |
| 10 | 9 | 68.92% | 0.3723 |
| 50 | 49 | 94.30% | 0.0587 |
| 100 | 99 | 97.29% | 0.0275 |
| 200 | 199 | 98.69% | 0.0132 |

The loop prints **before** applying that step's update. After all 200 updates, the final displayed probability is **98.70%**.

These are repetitions of one example, not 200 different sentences and not evidence of broad language learning.

## 14. Training versus using the trained model

After training, the code does:

```python
user_input = "I drink"
final_probabilities = predict_probabilities(encode(user_input))
```

This runs encoding, weighted sums, and softmax. It does not calculate gradients or change weights.

| Training | Final prediction |
|---|---|
| Uses input and correct answer | Uses input and trained parameters |
| Measures loss and calculates updates | Calculates output probabilities |
| Changes weights and biases | Leaves weights and biases unchanged |

This prediction phase is often called **inference**. It just means using the model.

The final probabilities are approximately:

```text
i      0.43%
drink  0.43%
tea   98.70%
coffee 0.43%
```

Rounded displayed values may not add to exactly 100%; the unrounded probabilities do, up to floating-point precision.

## 15. Choosing the next word

The code selects the candidate with the greatest probability:

```python
predicted_id = max(
    range(len(VOCABULARY)),
    key=lambda i: final_probabilities[i]
)
```

This says: inspect IDs 0, 1, 2, and 3, and return whichever ID has the largest probability.

The `lambda` is a small function that tells `max()` what value to compare. A longer equivalent is:

```python
predicted_id = 0
for candidate_id in range(len(VOCABULARY)):
    if final_probabilities[candidate_id] > final_probabilities[predicted_id]:
        predicted_id = candidate_id
```

Tea has ID 2, so:

```python
predicted_word = VOCABULARY[2]  # 'tea'
```

The final printed sentence is assembled with string concatenation:

```python
user_input + " " + predicted_word
```

Result:

```text
I drink tea
```

The model predicts only **tea**. The original “I drink” is preserved and the predicted word is appended by ordinary Python code.

We choose the maximum, rather than randomly drawing a word according to the probabilities. Running the unchanged program again produces the same result.

## 16. Where are “with” and “sugar”?

They are not in our vocabulary, not in our training sentence, and not possible outputs of this program.

We have not taught the model how to continue after “i drink tea”. That is why the script stops after one prediction.

If you fed the completed sentence back into this model, it would calculate another set of probabilities, but we have no training evidence that its continuation would be useful. A calculation producing an answer does not mean it has learned the task we want.

## 17. Is “tea” hardcoded?

Tea is explicitly provided as the **training answer**, just as labelled examples provide correct answers in supervised learning.

The prediction function does not contain a rule such as:

```python
if text == "I drink":
    return "tea"
```

It calculates scores from learned numbers. Its output becomes tea because training increased tea's score relative to the other scores.

But the task is deliberately trivial: one input and one answer. A lookup table could solve it more simply. We use training here to expose the mechanics, not because it is the best engineering solution for one sentence.

## 18. Other Python syntax you will encounter

### `def` and `return`

`def encode(text):` defines a reusable function. `return numbers` sends its result back to the caller.

For example, `input_numbers = encode(training_input)` stores the returned list in `input_numbers`.

### `[0.0] * len(VOCABULARY)`

`len(VOCABULARY)` is 4. Repeating `[0.0]` four times produces `[0.0, 0.0, 0.0, 0.0]`.

### The weights list comprehension

```python
weights = [[0.0] * len(VOCABULARY) for _ in VOCABULARY]
```

This creates a fresh four-number row for each output word. `_` means the loop's item is not otherwise needed.

Avoid replacing this with `[[0.0] * 4] * 4`: that repeats references to the same inner list, so changing one row can unexpectedly change the others.

### `+=` and `-=`

```python
score += amount
```

means `score = score + amount`.

```python
biases[output_id] -= adjustment
```

means subtract the adjustment from that stored bias.

### `.append()`

`scores.append(score)` adds one calculated score to the end of the list.

### `zip()`

```python
for word, probability in zip(VOCABULARY, probabilities):
```

Pairs each word with its corresponding probability. `zip()` stops when the shortest input ends; here both lists have four entries.

### Conditional expressions

```python
target = 1.0 if output_id == correct_word_id else 0.0
```

means: use 1.0 for the correct candidate, otherwise use 0.0.

### Formatted strings

```python
f"{probability:7.2%}"
```

Displays a proportion as a percentage with two decimal places. The width of 7 adds spacing for alignment. Formatting does not change the stored number.

`{word:6s}` aligns a string in a field of width 6. `{loss:.4f}` displays four decimal places. `{training_input!r}` displays the string's representation, including quotes.

`\n` starts a new printed line.

### `if __name__ == "__main__":`

When you run `python code.py`, Python sets `__name__` to `"__main__"`, and this block calls `main()`.

If another Python file imports this file, the block does not automatically run training. Some top-level setup still runs, including creating the vocabulary and initial weights.

### Where are the trained numbers saved?

They are kept in the `weights` and `biases` lists in memory while the program runs. The script does not save them to a model file.

Each new command-line run starts from zero and trains again. Calling `main()` twice in the same imported Python process would instead continue from the existing global weights; the current file is designed for one command-line run at a time.

## 19. Small experiments, with no surprise extra data

Try these one at a time, restoring the original setting afterwards.

### Experiment A: Train for only one step

Change:

```python
training_steps = 1
```

The final tea probability should be about **31.03%**. Tea already has the largest probability, even though that probability is below 50%.

That is possible because it only needs to exceed each of the other three candidates to be the maximum.

### Experiment B: Turn off training updates

Change:

```python
learning_rate = 0.0
```

All weights and biases stay at zero, so all probabilities remain 25%.

The maximum-selection code chooses the first candidate in a tie, which is “i”. That would be a tie-breaking result, not a learned preference for “i”.

### Experiment C: Reduce the learning rate

Change:

```python
learning_rate = 0.01
```

For this example, updates become smaller and learning progresses more slowly over the same 200 steps. Bigger rates are not always better; in general, overly large steps can make optimisation unstable.

### Experiment D: Try different capitalisation

Set `user_input` to `"I DRINK"`. The result stays the same because `encode()` lowercases it.

### Experiment E: Demonstrate the order limitation

Set `user_input` to `"drink I"`. The result stays the same because this representation counts words and discards their order. This is a weakness, not evidence that the two phrases mean the same thing.

### Experiment F: Explicitly replace the training answer

Only if you deliberately want to change the training example, replace:

```python
TRAINING_SENTENCE = "i drink tea"
```

with:

```python
TRAINING_SENTENCE = "i drink coffee"
```

This is a replacement example, not hidden additional data. On a fresh run, coffee becomes the learned answer, with the corresponding approximately 98.70% probability under the same settings.

Some explanatory comments and final printed messages specifically describe tea in the original example. They are teaching text, not adaptive reporting; update them if you make this change. The prediction calculation itself uses the target extracted from `TRAINING_SENTENCE`.

## 20. What this result does and does not establish

This program demonstrates:

- Converting known words into numerical inputs.
- Calculating scores using weights and biases.
- Converting scores into probabilities using softmax.
- Measuring a classification loss.
- Updating parameters using gradients.
- Using trained parameters to select a next word.

It does not demonstrate:

- Understanding the meaning of drinking, tea, or coffee.
- Learning word order from the input representation.
- Generalising to unseen sentences.
- Predicting an appropriate answer for arbitrary known-word combinations.
- Generating a useful paragraph.
- The architecture of a transformer or full LLM.

With one repeated target, this model is strongly pushed towards tea. It can confidently predict tea even for inputs that were never trained. High probability here is not a guarantee of correctness outside our one example.

If you supply more known words, the count list still has four positions; the counts change. Unknown words raise an error. Accepting a longer input is different from understanding it.

## 21. A short self-check

Before moving on, see whether you can answer these:

1. Why is `[1, 1, 0, 0]` the input for “I drink”?
2. Why do all candidates start at 25%?
3. Why is tea's first score gradient negative?
4. Why do tea's active input weights increase when we subtract the gradient?
5. Why is coffee an allowed output but unlikely after training?
6. Why does this program stop after predicting tea?

Answers:

1. There is one “i”, one “drink”, zero “tea”, and zero “coffee” in the input.
2. All initial scores are zero, so their exponentials are equal.
3. Its probability is 0.25 and its target is 1, giving 0.25 - 1 = -0.75.
4. Subtracting a negative adjustment adds a positive amount.
5. It belongs to the vocabulary, but the only training target is tea.
6. We supplied no example teaching what should follow “i drink tea”.

The main calculation to remember is the actual first update:

```text
Input: [1, 1, 0, 0]
Initial P(tea): 25%
Correct target for tea: 1
Tea score gradient: 0.25 - 1 = -0.75
Active tea weights: 0 -> 0.075
Tea bias: 0 -> 0.075
New tea score: 0.225
New P(tea): approximately 31.03%
```

Every later step repeats these same kinds of calculations using the updated numbers.
