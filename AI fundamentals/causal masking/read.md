# Causal masking: preventing attention from seeing future tokens

Causal masking restricts which input positions a prediction can use. In a
causal next-token language model, each position may use itself and earlier
positions, but not later positions.

The training system knows the target so it can calculate an error. The forward
calculation producing that prediction must not be allowed to read the target
from a future position.

This document accompanies [code.py](code.py). All scores are explicitly chosen
teaching values. The program demonstrates a mask and attention softmax; it
does not train a neural network or generate a learned completion.

## The exact sentence and prediction tasks

Use the same sentence throughout:

```text
I drink tea with sugar
```

Treat each word as a token. Python positions start at zero:

| Position | Token |
| --- | --- |
| 0 | I |
| 1 | drink |
| 2 | tea |
| 3 | with |
| 4 | sugar |

The sentence provides these next-word tasks:

| Visible input | Target used to calculate loss |
| --- | --- |
| I | drink |
| I drink | tea |
| I drink tea | with |
| I drink tea with | sugar |

The target is shifted one position ahead of the final visible input token.
At the drink position, the model already knows drink; its task is to predict
tea. It must not use the future tea, with, or sugar positions to do so.

No target after sugar is supplied in this example. The code therefore does not
invent an end token or an extra training sentence. A full training dataset can
include end tokens, but they need to be explicitly represented.

## Why a mask is needed

Training implementations can process several positions from a sequence in
parallel. Although the complete text is in the training system, each position's
representation must depend only on its permitted prefix.

Without that restriction, the representation at drink could attend directly
to the following tea position. This leaks information about the answer. A low
training error obtained this way would not reflect the actual generation task,
where the next token has not been supplied yet.

If a program processes only a prefix such as “I drink”, the later tokens are
absent altogether. The full-sequence illustration makes the restriction
visible when those later positions are present in the training computation.

## The complete causal mask

Rows represent query positions: the positions gathering information. Columns
represent key/value positions: the positions that may supply information.

| Query / key | I | drink | tea | with | sugar |
| --- | --- | --- | --- | --- | --- |
| I | YES | NO | NO | NO | NO |
| drink | YES | YES | NO | NO | NO |
| tea | YES | YES | YES | NO | NO |
| with | YES | YES | YES | YES | NO |
| sugar | YES | YES | YES | YES | YES |

The rule is:

```text
Allow attention when key_position <= query_position.
Block attention when key_position > query_position.
```

The diagonal is allowed because each position already contains its current
token. The next-token target is the following token, not the current one.

The mask is based on position, not word identity. If tea occurred earlier in
a longer input, that earlier occurrence could be attended to. Blocking the
future tea position is not a permanent ban on the word tea.

## Where the numerical scores come from

For the query at drink, use exactly these illustrative attention scores:

```text
I: 1     drink: 2     tea: 5     with: 3     sugar: 4
```

These numbers are supplied directly to isolate the masking operation. They
are not calculated from a hidden model or training dataset.

In scaled dot-product attention, scores are ordinarily calculated from query
and key vectors and divided by the square root of the key dimension. Here we
treat the supplied numbers as already scaled. No additional square-root
division is needed in this demonstration.

Without a mask, softmax over all five scores would give approximately:

| Input position | Unmasked attention |
| --- | ---: |
| I | 1.1656% |
| drink | 3.1685% |
| tea | 63.6409% |
| with | 8.6129% |
| sugar | 23.4122% |

Most attention would go to future positions in this chosen example, including
the target position tea. These are attention coefficients, not next-word
output probabilities.

## Apply the mask before softmax

Replace forbidden scores with negative infinity:

```text
Original: [1, 2, 5, 3, 4]
Allowed:  [YES, YES, NO, NO, NO]
Masked:   [1, 2, -infinity, -infinity, -infinity]
```

Conceptually this is the same as adding a mask containing zero for allowed
positions and negative infinity for forbidden positions.

Softmax uses exponentials. Negative infinity is useful because:

```text
exp(-infinity) = 0
```

Merely replacing forbidden scores with zero would be wrong: exp(0)=1, so those
positions would retain positive attention.

## Calculate the masked probabilities

For the two permitted scores:

```text
exp(1) = 2.718281828...
exp(2) = 7.389056099...
exp(-infinity) = 0

Total = 2.718281828 + 7.389056099 + 0 + 0 + 0
      = 10.107337927...
```

Normalize by that total:

```text
Attention to I     = 2.718281828 / 10.107337927
                   = 0.268941421... = 26.8941%

Attention to drink = 7.389056099 / 10.107337927
                   = 0.731058579... = 73.1059%

Attention to tea   = 0%
Attention to with  = 0%
Attention to sugar = 0%
```

The coefficients add to 100%. The masked positions contribute nothing to the
attention mixture at drink.

The program also changes the forbidden tea score to 1,000,000 as an explicit
experiment. Its attention remains zero because the mask replaces that score
before softmax. This demonstrates that the rule blocks a position regardless
of how attractive its original attention score was.

## Stable softmax in the code

Exponentials of very large positive numbers can overflow. Subtracting the same
constant from every score leaves softmax mathematically unchanged.

After masking, the maximum permitted score is 2. Subtract it:

```text
[1, 2, -infinity, -infinity, -infinity] - 2
= [-1, 0, -infinity, -infinity, -infinity]

Exponentials = [0.367879441..., 1, 0, 0, 0]
Total = 1.367879441...

I attention = 0.367879441 / 1.367879441 = 0.268941421...
drink attention = 1 / 1.367879441 = 0.731058579...
```

These are the same probabilities. The implementation finds the maximum after
masking, so even a huge forbidden score does not affect numerical scaling.

Every valid row of this causal mask permits at least the current position.
The helper explicitly rejects a mask that excludes every position because
there would be no valid distribution to normalize.

## How can the model output tea if tea is masked?

Input-position attention and output-vocabulary selection are separate steps.

```text
Visible input positions: I, drink
    → attention mixes information from those positions
    → further neural-network processing
    → scores over output vocabulary
    → output probabilities, including a probability for tea
```

Tea remains a valid output token. The rule only prevents reading the future
tea input position while calculating this prediction.

After the forward calculation, the training system compares the output
probabilities with target tea and calculates a loss. The target is permitted
to influence the loss and gradients; it must not leak into the input used to
make that same prediction.

The script stops at attention probabilities. It has no learned value vectors,
output projection, or training loop, so it does not pretend that the mask alone
has learned to predict tea.

## Causal masking compared with top-k

Both can involve exclusions, but they act on different objects:

| Mechanism | What is restricted | Purpose |
| --- | --- | --- |
| Causal mask | Input positions available to attention | Prevent information from future positions entering a prediction |
| Top-k sampling | Output tokens eligible for the current sample | Select among a limited set of likely next tokens |

The causal mask is a fixed position rule in this example. It does not rank
positions by likelihood, update weights, or determine the correct output.
Top-k acts on the final output distribution rather than on access to input
positions.

## Reading the Python

- `TOKENS` contains our five words in order.
- `SCORES` contains the chosen, already-scaled attention scores for drink.
- `causal_mask(length)` builds the triangular table using position comparisons.
- `masked_softmax(scores, allowed)` replaces blocked scores and calculates a
  numerically stable probability distribution.
- `math.inf` represents positive infinity; `-math.inf` represents negative infinity.
- `zip(...)` pairs corresponding entries from several lists.
- `enumerate(...)` supplies each list item's zero-based position and value.
- `TOKENS[:position + 1]` takes the prefix through the current position.
- `TOKENS[position + 1]` selects the next token as the target.
- `main()` displays the mask, target pairs, numerical comparison, and the
  large-forbidden-score experiment.

In this code, `True` means allowed and `False` means blocked. Different
libraries may use different Boolean mask conventions, so check the definition
when translating this example into another implementation.

## Reading and running

Read `read.md` in GitHub or a Markdown preview. It is a document, not a program.
To reproduce the calculations, open a terminal in this folder and run:

```sh
python code.py
```

Python 3 is sufficient. No packages, internet connection, API keys, GPU, or
downloaded model are required. The script prints results and exits.

## Small experiments

Change one thing at a time:

- Change the forbidden tea score from 5 to another finite number. The masked
  result at drink stays unchanged.
- Change the permitted I score from 1 to 2. The permitted scores tie, so I and
  drink each receive 50% attention; future positions still receive zero.
- Pass the first mask row to `masked_softmax`. Only I is allowed, so it receives
  100% attention, regardless of the other finite scores.
- Inspect the tea row of the mask: tea may use I, drink, and tea when predicting
  the following word with. The earlier drink row still cannot use tea.

Using the supplied score vector with a different row is a mathematical mask
experiment only. In a real model, each query position calculates its own scores.
