# Dense quantization: measured sensitivity and optimized allocations

**Qwen3.8-27B (dense FFN + hybrid attention). Measured on thor, wikitext-2,
n_ctx 512, 200 chunks, one shared imatrix, KL against BF16 reference logits.
BF16 PPL 6.7858 ± 0.0739.**

## Headline

Allocation only pays where there is damage to redistribute. **The crossover is
~15.2 GB.** Ship the measured allocation below it; use flat above it.

Every optimized arm came out a little smaller than its control, which biases the
raw comparison. The fair comparison interpolates the measured flat curve to the
optimized arm's exact size:

| tier | size | KL (measured alloc) | KL (flat, same size) | advantage |
|---|---|---|---|---|
| Balanced | 17.653 GB | 0.011211 | 0.009832 | **−14.0%** |
| Compact | 15.174 GB | 0.030468 | 0.030705 | **+0.8%** (break-even) |
| **Mini** | 13.492 GB | 0.049020 | 0.065369 | **+25.0%** |
| **Nano** | 10.789 GB | 0.121640 | 0.156226 | **+22.1%** (extrapolated) |

The flat curve those are measured against (same setup, same imatrix, same
reference logits):

| size | flat KL |
|---|---|
| 22.082 GB (uniform Q6_K) | 0.002111 |
| 17.640 GB | 0.009877 |
| 15.455 GB | 0.026947 |
| 13.673 GB | 0.061664 |
| 11.025 GB | 0.144783 |

That column is the whole explanation: the Q4 band is nearly lossless, so there
is nothing to move around, and a clever allocation only loses by mispricing.
By Mini there is ~6x the damage, and moving it pays ~25%.

**Ship Mini and Nano only.** Compact is a coin flip and carries no differentiation
claim; Balanced and above are actively worse than the shelf's flat recipe.

Note the gain is **not monotonic in damage** — it peaks at Mini (+25.0%) and
falls slightly at Nano (+22.1%). The floors are part of that: `token_embd`
bottoms out at IQ2_S (see below) and the low-bit type menu runs thin. Do not
assume a Micro tier extends the trend; it would need measuring.

## The sensitivity curve

Perturbation sweep: baseline everything Q6_K (22.08 GB, KL 0.002111), then drop
one group to Q3_K and measure ΔKL. The governing quantity is **ΔKL per GB
saved** — the exchange rate.

| group | GB saved | ΔKL | **ΔKL/GB** |
|---|---|---|---|
| token_embd | 0.496 | 0.001146 | **0.00231** |
| ffn (mid layers) | 4.595 | 0.025957 | 0.00565 |
| ffn_gate | 2.228 | 0.013274 | 0.00596 |
| ffn_up | 2.228 | 0.017950 | 0.00806 |
| linattn_gate | 0.589 | 0.005181 | 0.00880 |
| ffn_down | 2.228 | 0.020502 | 0.00920 |
| linattn_qkv | 0.983 | 0.010582 | 0.01077 |
| linattn_out | 0.589 | 0.008681 | 0.01474 |
| ffn (edge layers) | 1.044 | 0.015481 | 0.01483 |
| fullattn | 0.655 | 0.010371 | 0.01583 |
| output | 0.496 | 0.017489 | **0.03526** |

**15.3× spread.** Precision is not fungible on this architecture.

Three results worth keeping:

- **`output` is 15.3× dearer than `token_embd` at identical shape** (both
  5120×248320). Every shelf quant spaces them one step apart (Q4_K / Q6_K); the
  real gap is far wider. This is the largest single lever on the model, in a
  tensor the MoE configs never emitted at all.
- **FFN edge layers are 2.63× dearer per byte than middle layers.** The
  layer-position gradient — APEX's signature idea — *is* real on dense. The
  earlier null came from testing it in a band with no damage to redistribute.
- **The three linear-attention tensors are not interchangeable**: `linattn_gate`
  0.00880 vs `linattn_out` 0.01474, a 1.7× spread. This explains the earlier
  `dense_hybrid` failure — it cut all three uniformly, including the dear one,
  to fund pinning full-attention (0.01583, genuinely dear; that half was right).

## Two limits found the hard way

**The exchange rates do not extrapolate across precision.** They are averaged
over a large Q6_K→Q3_K interval, but sensitivity is convex — the first bit
removed is cheap, the last is expensive. A group that looks cheap over that
interval may already be in its steep region at Q4. This is why the measured
allocation *lost* at the Q4 band, and why `mini-v2` (which bought `output`
Q8_0 by cutting an already-Q3_K FFN middle to IQ3_XXS) came out **6.1σ worse
than `mini` v1**. Use the rates to rank groups, not to price arbitrary moves.

**`token_embd` cannot be compressed below the imatrix floor.** It is an
embedding lookup (`get_rows`), not a matmul, so `llama-imatrix` records nothing
for it — the imatrix has **zero** entries for `token_embd`. Very-low-bit types
that require imatrix data abort at quantize time (rc=133). IQ2_S works, IQ2_XXS
does not. So the cheapest group in the table cannot be fully cashed in. Same
class of failure as the MTP `blk.64` bail in llama.cpp PR #23476.

## Recommended settings

**Above ~15.2 GB: use a flat per-role allocation.** The shelf recipes are at or
near the optimum there; a sensitivity-driven allocation is measurably worse
(-14.0% at Balanced). Do not ship an APEX-Dense tier above the crossover — it
would be the same recipe unsloth and bartowski already publish, with no reason
for anyone to prefer it.

**Compact (~15.2 GB) is the crossover and is break-even (+0.8%).** Not worth a
differentiation claim either way.

**At Mini (~13.5 GB) — `mini` v1, the best measured allocation:**

| group | type |
|---|---|
| token_embd | IQ2_S (floor — no imatrix coverage) |
| output | Q6_K |
| fullattn / linattn_out | Q5_K |
| linattn_qkv | Q4_K |
| linattn_gate, ssm_alpha/beta | Q3_K |
| FFN edge (0–4, 59–63) | Q4_K / Q4_K / Q5_K |
| FFN near | Q3_K / Q3_K / Q4_K |
| FFN mid (10–53) | IQ3_XXS / Q3_K / Q3_K |

→ 13.492 GB, PPL 7.0202, KL 0.049020 (flat control: 13.673 GB, KL 0.061664)

**At Nano (~10.8 GB)**: same shape pushed down — see `nano` in
`scripts/generate_opt_config.py`. 10.789 GB, PPL 7.4122, KL 0.121640 (flat
control: 11.025 GB, KL 0.144783).

Generate with `scripts/generate_opt_config.py --level mini|nano`.

## What is not measured here

Speed. All of this is quality-per-byte; throughput effects of these mixes
(kernel selection per quant type, cache behaviour) would need a separate A/B
with the box quiet under one lock.

Whether these allocations transfer to other dense models. The exchange rates
are specific to this architecture — in particular the linear/full attention
split. Re-running the sweep on a new model is ~3.5 h and is the honest way to
port this, rather than assuming the ranking holds.
