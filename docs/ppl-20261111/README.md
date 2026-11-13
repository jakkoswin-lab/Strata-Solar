# Perplexity: IQ3_XXS against IQ2_XXS, Solar-Open-100B

Measured 2026-11-11 with `strata-glm4 --ppl` (this repo) and cross-checked with llama.cpp's `llama-perplexity`.  Both
quants saw the **same token ids in the same order**, and the perplexity is the model's own cross-entropy over the
scored positions (`exp(mean(-log p(token_t)))`).

The text was one chapter of a Chinese web novel and its Korean translation (the two languages this model is tuned for).
**The text itself is not included here** - it is a translator's work, not ours.  The numbers below are what the
comparison produced; any text of similar length reproduces them through `--tokens-file <ids>`.

| text | tokens | IQ3 nll | IQ2 nll | IQ3 ppl | IQ2 ppl | ratio | top-1 agreement |
|---|---|---|---|---|---|---|---|
| Chinese source | 399 | 2.7711 | 3.1375 | 15.976 | 23.046 | 1.443x | 274/399 = 68.7% |
| Korean translation | 399 | 3.3936 | 3.8142 | 29.774 | 45.338 | 1.523x | 262/399 = 65.7% |

nll is nats per token; bits/token = nll / ln 2 = zh: 3.998 -> 4.526, ko: 4.896 -> 5.503.

## Cold start against steady state

The first two positions have no context at all, and IQ2 is catastrophic there (+13.1 and +10.2 nats on the Chinese
text), which drags the headline ratio up.  Dropping the first three positions - what a real request always has -
gives the numbers to quote:

| text | from position | IQ3 ppl | IQ2 ppl | ratio | bits/token |
|---|---|---|---|---|---|
| Chinese source | 0 | 15.976 | 23.046 | 1.443x | +0.529 |
| Chinese source | 3 | 15.266 | 20.682 | 1.355x | +0.438 |
| Korean translation | 0 | 29.774 | 45.338 | 1.523x | +0.607 |
| Korean translation | 3 | 28.696 | 39.888 | 1.390x | +0.475 |

## What it says

IQ2_XXS is not usable for this workload: +35-39% perplexity on the steady-state positions, and 31-34% of the positions
pick a different top-1 token.  IQ3_XXS is the production quant.  (The current production numbers, measured on a
1,876-token chapter, are in `docs/SOLAR_PORT.md`: IQ3_XXS 19.37 ppl, IQ2_S 21.54, IQ2_XXS 26.54.)
