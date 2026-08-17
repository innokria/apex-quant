# APEX-Dense allocation experiment — Qwen3.8-27B

**Result: both levers failed *in the Q4_K_M band*. Do not ship an APEX-Dense
ladder built on them at that size.**

## Scope of this null result — read before citing it

This tested one size band: **~17.5 GB, the Q4_K_M / UD-Q4_K_XL class**. In that
band every arm landed within **0.008–0.012 KL of BF16** — the model is close to
lossless there, so there is very little damage available to redistribute. "No
allocation helped" therefore partly means "there was nothing to win."

The band was chosen because it is where the shelf competition sits. That is a
positioning rationale, not a measurement one, and it is the weakest part of this
experiment's design. Allocation should matter most where quality is scarce —
the Compact/Mini band (Q3_K/IQ3, ~12–14 GB), which is **untested**.

So: these levers are dead at Q4. They are *unproven*, not disproven, lower down.
See `NEXT_EXPERIMENT.md` — the follow-up measures sensitivity directly instead
of guessing allocations.

## What was tested

Whether APEX's ideas transfer from MoE to a dense-FFN model. Three arms at
**matched size**, differing only in how bits are allocated, all sharing one
imatrix (calibration_v1.3, 200 chunks) so allocation is the only variable:

| arm | allocation |
|---|---|
| `dense_flat` | control — flat per-role, no layer gradient (replicates unsloth's UD recipe) |
| `dense_grad` | control + FFN layer-position gradient |
| `dense_hybrid` | grad + full-attn pinned Q8_0 / linear-attn cut to Q4_K |

`dense_hybrid_quality` is the same allocation in the Q5/Q6 band and is **not**
size-matched — it is a sanity check, not an arm.

Measured on thor (NVIDIA Thor, sm_110), wikitext-2-raw, n_ctx 512, 200 chunks.
KL is against BF16 reference logits. **BF16 baseline: PPL 6.7858 ± 0.07388.**

## Results

| arm | size GB | PPL | KL mean | KL median | KL 99.9% |
|---|---|---|---|---|---|
| dense_flat (control) | 17.64 | 6.8114 ± 0.0743 | **0.009877 ± 0.000152** | 0.004050 | 0.3766 |
| dense_grad | 17.55 | 6.8117 ± 0.0742 | 0.009926 ± 0.000166 | 0.004085 | 0.3433 |
| dense_hybrid | 17.48 | 6.8097 ± 0.0741 | 0.011772 ± 0.000193 | 0.004803 | 0.4454 |
| dense_hybrid_quality | 19.78 | 6.7904 ± 0.0738 | 0.007762 ± 0.000149 | 0.003096 | 0.2870 |

**PPL does not discriminate here.** All arms sit within ±0.074 of each other and
of BF16; the spread between arms is 0.002–0.026. Only KL separates them, which
is why it was measured.

## Verdict per lever, against the control

| lever | ΔKL mean | σ | z | verdict |
|---|---|---|---|---|
| FFN layer-position gradient | +0.000049 | 0.000225 | **+0.2** | **null** — no measurable effect |
| hybrid-attention split | +0.001895 | 0.000246 | **+7.7** | **harmful** — clearly worse |

**The FFN layer-position gradient does nothing on this model.** APEX's signature
move — the lever that carried the entire MoE result — is indistinguishable from
flat allocation at 0.2σ. It buys 0.09 GB, which is noise.

**The hybrid-attention split actively hurts.** KL is ~19% worse at 7.7σ, with
median and 99.9% tails both worse, while PPL is unchanged. The hypothesis was
that the 48 linear-attention layers (20% of params) could absorb a cut to Q4_K
to pay for pinning the 16 full-attention layers (6%) to Q8_0. The data says the
opposite: **linear-attention tensors are more precision-sensitive than the
budget share suggests**, and cutting them costs more than pinning full-attn
recovers.

## Why this is a plausible outcome, not a bug

The MoE result rested on sparsity: routed experts are ~90% of params but only
k/N fire per token, so low precision there is nearly free. On this model the
FFN is 62.6% of params and **every weight is on the critical path for every
token**. There is no redundancy for a gradient to exploit. Meanwhile the shelf
quants (unsloth UD, bartowski, ggml-org, ubergarm) already allocate sensibly
per role — the control *is* unsloth's recipe, and it is hard to beat.

Sanity checks that this measures what it claims:
- size model predicted all four arms to within **0.08%** (17.627→17.64,
  17.540→17.55, 17.466→17.48, 19.772→19.78)
- arms are size-matched to 0.16 GB (0.9%), so quality differences are not size
- one imatrix shared by all arms; one `flock` held across the whole sweep

## Recommendation

- **Do not publish an APEX-Dense tier ladder on these levers at the Q4 band.**
  There is no quality story to tell against the existing shelf quants there.
- The dense/hybrid generator mode, the size estimator and the tests are worth
  keeping — they made this measurable and cost little.
- **Next: measure sensitivity instead of guessing it** (`NEXT_EXPERIMENT.md`).
  One perturbation sweep yields the per-group precision-sensitivity curve for
  the whole model, and would have flagged linear-attention as sensitive before
  an arm was spent assuming the opposite.
- The strongest data-driven hypothesis from this run is the **inverse** of what
  was assumed: linear-attention may deserve MORE bits, paid for out of the FFN
  middle. The sweep tests that directly rather than as a one-off arm.

## On the MTP-imatrix experiment (revised down)

Previously listed here as "worth running". That was carried on momentum. Two
corrections:

- `quantize_q8_0()` contains `(void)quant_weights; // not used` — **Q8_0 ignores
  imatrix entirely**. With the drafter pinned at Q8_0, as APEX-MTP does today,
  imatrix on the MTP head is a no-op for our own repos.
- The useful form of the question is therefore **drafter at Q6_K with imatrix
  vs Q8_0 without** — Q6_K does consume `quant_weights`, and is 6.5625 vs
  8.5 bpw. That is a three-quant test varying only `blk.64`, not a matrix.

llama.cpp PR #23476 is still worth landing on its own merit: without it, low-bit
i-quants of MTP-bundled models bail at quantize time. That value is independent
of any measurement.
