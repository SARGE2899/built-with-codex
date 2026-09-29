# Top-k sampling: choosing from the most likely words

This lesson answers your question:

> If tea and coffee are the top two words and we use top-k = 2, can sugar still appear?

**For that prediction step, no.** Sugar is removed from the candidates used
for sampling. This is a rule enforced by the selection code, not just a claim
that sugar is unlikely.

Read this document alongside [code.py](code.py). The script uses only Python's
standard library and has comments explaining the calculations and syntax.

## 1. Keep the same example

Our input is:

```text
I drink
```

We use exactly these illustrative next-word probabilities:

| Candidate word | Decimal probability | Percentage |
| --- | ---: | ---: |
| tea | 0.70 | 70% |
| coffee | 0.25 | 25% |
| sugar | 0.05 | 5% |
| Total | 1.00 | 100% |

Where did these numbers come from? We chose them for the explanation in our
conversation. This script supplies them directly. It does not train a neural
network or calculate them from the words “I drink.” There is no hidden dataset.
The prompt is displayed to show which prediction we are illustrating.

In a real language model, the network would calculate scores for its output
vocabulary, and softmax would turn those scores into probabilities. This lesson
starts with those probabilities already available and focuses on how to select
the next token.

For simplicity, each whole word represents a token here. Actual model tokens
can also represent parts of words or punctuation.

## 2. Why not simply select tea?

We can. Choosing the highest-probability word is called greedy selection.
For these numbers, it always chooses tea.

Sampling instead makes a random choice using the probabilities. Without
filtering, tea has a 70% chance, coffee 25%, and sugar 5%. Sugar can therefore
be selected even though it is the least likely word.

Top-k restricts the candidates before that random choice. It combines a
deterministic filtering step with a sampling step.

## 3. What does k mean?

**k is the number of highest-probability candidates we keep.**

It is a setting chosen by the application or user. It is not a learned weight,
an error metric, or the number of words we want in the completed sentence.

For k=2:

```text
Rank 1: tea       70%  -> keep
Rank 2: coffee    25%  -> keep
Rank 3: sugar      5%  -> exclude for this selection
```

We have not deleted sugar from the vocabulary or changed the original model
probabilities. We are creating a separate distribution for this selection.

## 4. Why do we divide by 95?

After excluding sugar, the remaining probabilities total:

```text
70% + 25% = 95%
```

We need the probabilities used for the random choice to add to 100%. Divide
each retained probability by the retained total:

```text
P(tea after filtering)    = 0.70 / 0.95 = 70/95 = 14/19
                         = 0.736842105... = 73.6842105...%

P(coffee after filtering) = 0.25 / 0.95 = 25/95 = 5/19
                         = 0.263157894... = 26.3157894...%

P(sugar after filtering)  = 0
```

This operation is called **renormalization**. The retained probabilities now
sum to 1:

```text
14/19 + 5/19 = 19/19 = 1
```

The ratio between tea and coffee remains the same: 70:25 equals 14:5.
We are not distributing the removed 5% equally between them. We scale both
retained probabilities by the same factor, 1/0.95.

In implementations that filter scores before softmax, excluded scores can be
masked out instead. For a fixed set of retained tokens, that gives the same
mathematical distribution as retaining and renormalizing their probabilities.

## 5. How does Python make the random choice?

Imagine a line from 0 to 1 divided into intervals:

```text
0                         0.736842...                  1
|------------ tea ------------|-------- coffee -------|
```

The script draws a random number greater than or equal to 0 and less than 1.

| Illustrative draw | Selected word | Reason |
| --- | --- | --- |
| 0.20 | tea | Below 0.736842... |
| 0.70 | tea | Below 0.736842... |
| 0.90 | coffee | At least 0.736842... and below 1 |

There is **no interval for sugar**. It cannot be selected from this filtered
distribution, even if the draw is close to 1.

The demonstration uses random seed 7. A seed makes the sequence of random
draws reproducible for this run; it does not change the filtering rule. The
first draw is approximately 0.323833, so the displayed completion is:

```text
I drink tea
```

## 6. What do the repeated trials demonstrate?

The program repeats the same next-word selection 10,000 times with fresh
random draws. Each trial starts from the same fixed distribution. It does not
append all those words into one sentence, train a model, or change weights.

You should see tea selected roughly 73.68% of the time and coffee roughly
26.32%. The exact observed frequencies need not equal those probabilities.
Random sampling does not promise exactly 14 teas and 5 coffees in every
19 draws.

Sugar's count is exactly zero because it is excluded. The guarantee comes
from the code sampling only the retained dictionary. Seeing zero sugar in a
finite experiment alone would not prove impossibility if sugar still had a
small positive probability.

## 7. Compare k=1, k=2, and k=3

All rows below start from our same original 70%, 25%, and 5% probabilities:

| Setting | Tea after filtering | Coffee after filtering | Sugar after filtering |
| --- | ---: | ---: | ---: |
| k=1 | 100% | 0% | 0% |
| k=2 | 73.6842% | 26.3158% | 0% |
| k=3 | 70% | 25% | 5% |

With k=1, only tea remains. Sampling from one candidate is effectively greedy
selection here. With k=3, all candidates remain, so the original distribution
is preserved.

In this script, k must be an integer between 1 and the number of candidates.
The code rejects invalid values rather than silently changing their meaning.
If probabilities are tied at the cutoff, our implementation keeps dictionary
insertion order and retains exactly k candidates. Other implementations may
have different tie-handling conventions. Our main example contains no ties.

## 8. Can sugar appear at a later step?

Yes, it can become eligible at a later step if the model gives it a top-k
ranking then. Top-k is applied separately to each step's predictions.

For example, after choosing tea, the context becomes:

```text
I drink tea
```

A real model would calculate new probabilities for this new context. We have
not supplied those probabilities, so this script does not invent a continuation
or claim which word would come next.

The precise guarantee is: **with the current probabilities and k=2, this
selection cannot choose sugar.** It is not a permanent prohibition against
sugar anywhere in a generated response.

## 9. How does this relate to temperature and training?

| Concept | Role |
| --- | --- |
| Learned weights | Produce scores from the input during the model's forward pass |
| Learning rate | Controls the size of parameter updates during training |
| Temperature | Rescales scores before softmax, changing how concentrated sampling probabilities are |
| Top-k | Restricts sampling to the k highest-ranked candidates |

Our lesson implements only top-k. It does not apply temperature or update
weights. For a fixed score vector, positive temperature changes probabilities
but preserves score ranking. Filtering with top-k still excludes candidates
outside the retained set.

Top-k does not understand correctness. If sugar had a top-two probability, it
would remain eligible with k=2. Restricting candidates cannot guarantee that
the selected sentence is sensible or factually correct.

## 10. Follow the code in order

1. `PROBABILITIES` stores our explicitly supplied example numbers.
2. `top_k_distribution` checks the input, sorts candidates, keeps the top k,
   and divides by their total probability.
3. `sample_word` draws from only the resulting candidates.
4. `main` displays the distribution, one completion, repeated trials, and the
   comparison between k settings.

Useful Python syntax used in this lesson:

- A dictionary such as `{"tea": 0.70}` maps a key to a value.
- `.items()` supplies `(word, probability)` pairs from that dictionary.
- `lambda pair: pair[1]` tells sorting to look at each pair's probability.
- `reverse=True` sorts from largest to smallest.
- `ranked[:k]` takes the first k elements of a list.
- `.get(word, 0.0)` returns zero when that word is absent.
- `Counter` counts how many times each sampled word appears.
- `_` in the repetition loop means the loop number itself is not needed.
- `:.4%` displays a decimal probability as a percentage with four decimal places.
- `if __name__ == "__main__":` runs the demonstration when the file is
  executed directly, rather than when its functions are imported.

## 11. How to use the files

`read.md` is for reading in GitHub or a Markdown preview. You do not execute it.
To run the Python demonstration, open a terminal in this lesson's folder:

```sh
python code.py
```

Use Python 3. No additional packages, API keys, internet connection, or GPU
are required. The program prints its results and exits without saving a model.

## 12. Small experiments

Change one setting at a time:

- Set `K = 1`. Every selection must be tea for these probabilities.
- Set `K = 3`. Sugar becomes eligible with a 5% probability on each draw.
- Change `SEED`. Sampled counts may change, but the top-k distribution stays
  the same. With k=2, sugar remains impossible for this step.
- Set `TRIALS = 20`. The small sample's observed percentages may differ
  noticeably from the theoretical probabilities.

If you edit the input probabilities, keep them nonnegative and make sure they
add to 1. Any new values are an explicitly changed illustration, not a result
of training the script.
