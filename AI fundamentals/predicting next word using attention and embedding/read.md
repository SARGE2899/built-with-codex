# Predicting the next word using attention and embeddings

Read this guide alongside [code.py](code.py). It follows the exact input:

> I had bread this morning and I drink tea with _____

This lesson connects tokenisation, embeddings, position information, queries, keys, values, attention, output probabilities, and automatic weight updates. It uses actual numbers produced by the script, rounded for readability.

## 1. The answer and the training data are explicit

We assume **bread** is the intended training answer, based on the example discussed in the lesson. The sentence alone does not make bread the uniquely correct completion. Sugar, coffee, or other continuations might be intended in a different situation.

The entire training dataset consists of this ONE pair:

```text
Input:  i had bread this morning and i drink tea with
Target: bread
```

The blank is a placeholder, not an input token. The earlier occurrence of bread remains in the input because it was part of the supplied sentence. The withheld final answer is not appended to the input during training.

There are no hidden training sentences and no pretrained language model. All parameters start from reproducible small random numbers.

**Important result:** training this one example teaches a strong preference for bread. It does not prove that the model needs or understands the earlier bread clue. The script includes a prediction-only probe that exposes this limitation rather than concealing it.

## 2. Vocabulary: possible words, not answers by themselves

The requested vocabulary was:

```python
["i", "drink", "tea", "bread", "coffee", "sugar"]
```

The prompt contains five additional words. The script appends them, in the order first encountered:

```python
["i", "drink", "tea", "bread", "coffee", "sugar",
 "had", "this", "morning", "and", "with"]
```

| Word ID | Word |
|---:|---|
| 0 | i |
| 1 | drink |
| 2 | tea |
| 3 | bread |
| 4 | coffee |
| 5 | sugar |
| 6 | had |
| 7 | this |
| 8 | morning |
| 9 | and |
| 10 | with |

All **11 words** are possible outputs. Coffee and sugar are allowed candidates even though neither appears in the training input.

The model does not choose from a special list containing only sensible answers. It calculates a probability for every vocabulary entry, including words such as “had” and “and”.

An ID is just a lookup label. Bread's ID of 3 does not represent its importance or probability.

## 3. Input positions are different from vocabulary IDs

The sentence contains **10 token positions**, because “i” occurs twice:

| Position | Token | Vocabulary ID |
|---:|---|---:|
| 0 | i | 0 |
| 1 | had | 6 |
| 2 | bread | 3 |
| 3 | this | 7 |
| 4 | morning | 8 |
| 5 | and | 9 |
| 6 | i | 0 |
| 7 | drink | 1 |
| 8 | tea | 2 |
| 9 | with | 10 |

There will be **10 attention weights**, one per input position, and **11 output probabilities**, one per vocabulary word. These are two different distributions.

The tokenizer lowercases and extracts sequences of English letters with `re.findall(r"[a-z]+", text.lower())`. This is a deliberately small whole-word tokenizer. It removes the underscores and punctuation; it is not a modern subword tokenizer and will not preserve arbitrary languages, numbers, or punctuation.

## 4. Which numbers do people choose, and which numbers are learned?

We choose the model structure, vector size of two, initialisation method, training pair, loss definition, learning rate, and number of repetitions.

The program learns:

| Parameters | Shape | Number of adjustable values |
|---|---|---:|
| Embedding table | 11 words × 2 numbers | 22 |
| Query transform | 2 × 2 | 4 |
| Key transform | 2 × 2 | 4 |
| Value transform | 2 × 2 | 4 |
| Output weights | 11 candidates × 2 numbers | 22 |
| Output biases | 11 candidates | 11 |
| Total | | **67** |

All matrix entries begin between -0.5 and 0.5. Biases begin at zero. Seed 7 makes the initial random choices repeatable; it supplies no language knowledge.

The equations themselves do not get invented during training. Training updates the parameters inside the equations.

## 5. Look up embeddings

Each token ID selects a two-number row from the embedding table. For example, before training:

```text
bread embedding = [-0.442001,  0.007436]
with embedding  = [ 0.476255, -0.453417]
```

These are adjustable numbers, not hand-assigned meanings such as “food” or “drink”.

The two occurrences of “i” share one embedding row. They are the same vocabulary item at different positions.

Lookup is equivalent to multiplying a one-hot word vector by an embedding table, but directly selecting the row avoids multiplying by many zeros.

## 6. Add a small signal for position

To distinguish positions, this lesson adds a fixed illustrative position vector:

```text
position vector at position t = [t/20, -t/20]
input vector = embedding + position vector
```

For bread at position 2:

```text
position vector = [0.1, -0.1]
input = [-0.442001 + 0.1, 0.007436 - 0.1]
      = [-0.342001, -0.092564]
```

For with at position 9:

```text
position vector = [0.45, -0.45]
input = [0.476255 + 0.45, -0.453417 - 0.45]
      = [0.926255, -0.903417]
```

This position rule is fixed, not trained. It was chosen for this teaching model; it is not a claim about the specific positional system used by real LLMs. Position information makes order available, but this one-example training run does not demonstrate that the model learns a useful interpretation of order.

## 7. Calculate a query for the final position

We predict the next word using the current final position, **with**. Its query compares with keys for every position already in the sentence.

The initial query transformation is approximately:

```text
Q1 =  0.358468*x1 - 0.210391*x2
Q2 = -0.355745*x1 - 0.382208*x2
```

Substitute with's input vector:

```text
Q1 =  0.358468*0.926255 + (-0.210391)*(-0.903417)
   = approximately 0.522104

Q2 = -0.355745*0.926255 + (-0.382208)*(-0.903417)
   = approximately 0.015783
```

So:

```text
query = [0.522104, 0.015783]
```

Nothing in this formula explicitly says “look for bread”. The query is a numerical transformation of the input at the final position.

## 8. Calculate keys for all ten input positions

Each position uses the SAME learned key transformation:

```text
K1 = -0.191518*x1 + 0.316126*x2
K2 = -0.319274*x1 + 0.081600*x2
```

For the earlier bread token:

```text
K1 = (-0.191518)*(-0.342001) + 0.316126*(-0.092564)
   = approximately 0.036237

K2 = (-0.319274)*(-0.342001) + 0.081600*(-0.092564)
   = approximately 0.101639
```

Bread's key is `[0.036237, 0.101639]`. The script calculates and prints all other keys the same way.

## 9. Turn query-key comparisons into scores

For each available position:

```text
score = dot(query, key) / sqrt(2)
```

A dot product multiplies matching entries and adds them:

```text
bread score = (0.522104*0.036237 + 0.015783*0.101639) / sqrt(2)
            = approximately 0.014513
```

Why square root of two? Each query and key contains **two numbers**. The division is a scaling step before softmax. It is not based on the ten input tokens or eleven vocabulary words.

The attention method specifies this calculation. Training adjusts the query and key transformation weights that feed it.

## 10. First softmax: attention across input positions

The ten initial scaled scores are approximately:

```text
[-0.027983, 0.034456, 0.014513, -0.033396, 0.004817,
 -0.065851, -0.085549, -0.128660, -0.095217, -0.175051]
```

For each position, exponentiate its score and divide by the sum of exponentials for all ten positions.

For bread:

```text
attention_to_bread = exp(0.014513) / sum(exp(each of the ten scores))
                   = approximately 10.71%
```

| Position | Input word | Initial attention |
|---:|---|---:|
| 0 | i | 10.26% |
| 1 | had | 10.92% |
| 2 | bread | 10.71% |
| 3 | this | 10.21% |
| 4 | morning | 10.60% |
| 5 | and | 9.88% |
| 6 | i | 9.69% |
| 7 | drink | 9.28% |
| 8 | tea | 9.59% |
| 9 | with | 8.86% |

These percentages describe mixing coefficients across input positions. They are **not probabilities that these words will be the next output**. They do not constitute a guaranteed explanation of semantic importance either.

The two i positions have separate attention weights, even though they share one word embedding. Their positional inputs differ.

## 11. Calculate values: the information to mix

We use a learned value transformation rather than mixing the raw embeddings directly:

```text
V1 = 0.138913*x1 - 0.127602*x2
V2 = 0.047744*x1 - 0.437211*x2
```

Bread's value is:

```text
V1 = 0.138913*(-0.342001) + (-0.127602)*(-0.092564)
   = approximately -0.035697

V2 = 0.047744*(-0.342001) + (-0.437211)*(-0.092564)
   = approximately 0.024141
```

Each position supplies its own two-number value list.

| Item | Role |
|---|---|
| Query | Used for comparisons at the predicting position |
| Keys | Compared with the query to calculate mixing weights |
| Values | The numerical information mixed together |

All three are numerical representations, not words or English instructions.

## 12. Mix values using attention

For each of the two dimensions:

```text
context[j] = sum(attention[position] * value[position][j])
```

Bread contributes approximately:

```text
first component:  0.1071 * (-0.035697) = -0.003823
second component: 0.1071 *   0.024141  =  0.002585
```

That is only one contribution. Add the corresponding contributions from the other nine positions. The complete initial mixture is:

```text
context = [0.066073, 0.154098]
```

This intermediate list goes to the output calculation. It does not overwrite any word's embedding table entry.

## 13. Second softmax: probabilities across output words

Each of the eleven candidate output words has its own two weights and bias:

```text
output score = weight1*context1 + weight2*context2 + bias
```

Initially, bread's output weights are approximately `[-0.046816, -0.200233]`, and its bias is zero:

```text
bread score = (-0.046816)*0.066073 + (-0.200233)*0.154098 + 0
            = approximately -0.033949
```

After calculating all eleven scores, apply softmax over those eleven scores. Initially:

```text
P(next word = bread) = approximately 8.8114%
```

The highest initial probability belongs to **had**, about 9.6743%. Randomly initialised weights have not learned sensible next-word behaviour.

Compare:

| Distribution | What it ranges over | Initial bread value |
|---|---|---:|
| Attention | 10 input positions | 10.71% at bread's position |
| Output probabilities | 11 vocabulary choices | 8.8114% for bread |

Different calculations, different purposes, different numbers.

## 14. Target vector and squared error

We continue with **mean squared error** to connect this lesson to the previous one. Cross-entropy is the usual objective for categorical next-token prediction, but is not the loss implemented here.

In vocabulary order, bread's target is:

```text
[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
```

Bread gets 1 because it is the supplied correct answer for this example. All other candidates get 0.

```text
MSE = sum((probability[j] - target[j])^2 for all output words) / 11
```

The divisor is **11 outputs**, not 10 input positions or 2 embedding dimensions.

The initial MSE is approximately **0.08316447**. Its numerical scale differs from the earlier two-output example because the mean is over eleven terms. Comparing raw MSE across different output counts can be misleading.

## 15. Backpropagation follows the complete chain

Training follows these dependencies backwards:

```text
MSE
 -> output probabilities
 -> output scores and biases
 -> output weight matrix and context mixture
 -> attention coefficients and values
 -> attention scores
 -> queries and keys
 -> Q/K/V weight matrices
 -> the input word embeddings
```

Every adjustable parameter receives a derivative through the routes that use it. No human chooses which word should receive high attention on each update.

The two softmax operations use the same general derivative procedure, but with different incoming gradients and different list lengths.

For MSE, the derivative with respect to output probability j is:

```text
probability_gradient[j] = (2/11) * (probability[j] - target[j])
```

For softmax, if g is the incoming probability-gradient list:

```text
common = sum(probability[k] * g[k])
score_gradient[j] = probability[j] * (g[j] - common)
```

This includes the effect of one score on all probabilities. The simpler cross-entropy shortcut `probability - target` is not the correct score gradient for this MSE loss.

## 16. Two actual first-update examples

The learning rate is 0.3. Every update uses:

```text
new_parameter = old_parameter - 0.3 * gradient
```

### Bread's output bias

Initially it is zero, and its first gradient is approximately -0.01465592:

```text
new bias = 0 - 0.3*(-0.01465592)
         = approximately +0.00439678
```

The negative gradient causes the update to increase bread's bias.

### Bread's input embedding

The initial embedding is `[-0.442001075, 0.007435733]`. Its first gradient is approximately `[0.000029795, -0.000164398]`:

```text
new first value  = -0.442001075 - 0.3*(0.000029795)
                 = approximately -0.442010014

new second value = 0.007435733 - 0.3*(-0.000164398)
                 = approximately 0.007485053
```

These changes are much smaller than the bias change. An embedding update is not automatically positive just because its word is the target. Its direction depends on how that input embedding affects the loss through the entire network.

Both i positions contribute gradients to the same i embedding row. The program adds those contributions instead of overwriting one with the other.

All gradients are calculated before any parameters change, so the update is based on one consistent forward pass.

## 17. The computer repeats the updates

The loop visits the same example 2,000 times:

```text
Forward calculation -> loss derivatives -> update all parameters -> repeat
```

No new training data is introduced by repetition. No person edits the weights between steps.

Reported metrics are measured after each indicated update:

| Updates completed | P(bread) | MSE |
|---:|---:|---:|
| 0 | 8.8114% | 0.08316447 |
| 1 | 8.8517% | 0.08309089 |
| 10 | 9.2291% | 0.08240397 |
| 100 | 15.0672% | 0.07214228 |
| 500 | 75.4901% | 0.00600794 |
| 1000 | 87.7560% | 0.00149940 |
| 2000 | 92.8022% | 0.00051821 |

The script trains from scratch each time. It saves no trained model file and requires no internet or downloaded weights.

## 18. All final output probabilities

| Candidate word | Probability after training |
|---|---:|
| i | 0.6412% |
| drink | 0.7259% |
| tea | 0.7148% |
| bread | **92.8022%** |
| coffee | 0.7764% |
| sugar | 0.7069% |
| had | 0.7767% |
| this | 0.7106% |
| morning | 0.7095% |
| and | 0.7490% |
| with | 0.6868% |

The program selects the maximum-probability output, bread. It does not sample randomly.

You can therefore complete the displayed input as:

> I had bread this morning and I drink tea with bread

This is the chosen training completion, not a claim that it is the only natural or correct continuation.

## 19. Did attention learn to prioritise bread?

Not strongly in this run. Bread's attention weight ends around **10.59%**, while had receives around **10.79%**. The distribution is still fairly spread out.

Yet the output probability of bread reaches 92.80%. This illustrates why attention percentages and output probabilities must not be equated.

We also run this **prediction-only diagnostic**, without training on it:

```text
I drink tea with
```

There is no earlier bread token. Nevertheless, the model assigns bread about **93.0374%**.

The result shows that this single-example training run learned a strong bread preference. It does not establish that the earlier clue is necessary. In particular, the learned output bias provides a way to favour the same target regardless of input.

We keep this diagnostic visible rather than forcing attention to bread, manually boosting bread's score, or claiming the model understands context.

To study reliable context-dependent choices, a later experiment would need varied examples with different answers and tests on held-out contexts. That dataset is deliberately not silently added here.

## 20. Why no future-token mask appears in this code

The script computes just the attention output at the **last existing input position**. Every key/value position is already available to that position. No future token is supplied, so there are no future positions to mask in this calculation.

A full causal language model that predicts at every position during training needs to prevent earlier positions from seeing later tokens. That broader all-positions training arrangement is not implemented here.

The model also has no stacked transformer blocks, multiple attention heads, residual connections, layer normalisation, or feed-forward block. The attention mixture goes directly to a linear output layer so its mechanics remain inspectable.

## 21. Python reading map

| Function | Purpose |
|---|---|
| `tokenize()` | Extract lowercase words |
| `new_model()` | Initialise the 67 adjustable numbers |
| `dot()` | Multiply matching entries and add |
| `matvec()` | Apply a row-by-row weighted transformation |
| `transpose_matvec()` | Send gradients backwards through a matrix |
| `softmax()` | Convert a score list into probabilities |
| `forward()` | Look up embeddings, add positions, calculate attention and predictions |
| `loss()` | Measure mean squared error |
| `softmax_backward()` | Differentiate through either softmax |
| `backward()` | Calculate all parameter gradients |
| `update()` | Subtract learning-rate-scaled gradients |
| `show_forward()` | Print the intermediate numbers and all output probabilities |
| `main()` | Run the demonstration and training loop |

`enumerate()` supplies a position and an item. It assigns word IDs and iterates through token positions. `zip()` pairs corresponding entries. List comprehensions build result lists from repeated calculations.

`cache` is simply a dictionary of intermediate forward-pass results. It lets backpropagation reuse the inputs, queries, keys, values, and probabilities from that exact calculation.

The target ID is passed to the loss and backward functions. It is not passed to `forward()`. Prediction therefore uses the trained numerical parameters, not a hardcoded `return "bread"` rule.

## 22. Small experiments

- Set `LEARNING_RATE = 0.0`: weights do not change and predictions remain at their initial values.
- Set `TRAINING_STEPS = 1`: inspect the small first improvement instead of the final learned preference.
- Change the seed: start from different numbers and observe a different attention pattern and trajectory. You are not changing the labelled training pair.
- Change TARGET deliberately to sugar: this explicitly changes the supervision. Training will favour that answer even though bread appears earlier. The printed bread-specific diagnostics and the tables in this document describe the original configuration; update explanatory reporting if you change the target.
- Compare inputs with and without the earlier bread clause using `forward()`, without calling `update()`. This tests predictions, not additional training.

Adding vocabulary rows gives possible choices, not knowledge about how to use them. More sentence length means more positions to process with the same transformations; it does not require a new handwritten formula for each word.

## 23. Verification and interpretation

The script was run with its documented settings. All 67 analytical parameter gradients were checked against numerical finite-difference estimates at the initial state; the maximum absolute difference was below 0.00000001.

That check validates the backpropagation arithmetic. It does not validate language understanding or generalisation. The successful bread prediction is measured on the same example used for training.

The complete lesson is:

```text
Words -> IDs -> learned embeddings + fixed positions
      -> query, keys, values
      -> attention scores / sqrt(2)
      -> softmax over input positions
      -> weighted mixture of values
      -> output scores
      -> softmax over vocabulary words
      -> predicted next word
```

During training, squared error supplies feedback through that entire chain. During prediction, the same forward calculations run with fixed learned parameters.
