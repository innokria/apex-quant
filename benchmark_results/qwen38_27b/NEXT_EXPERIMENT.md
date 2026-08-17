# Next experiment: per-group precision sensitivity sweep (Qwen3.8-27B)

**Measure where the bits actually matter, instead of guessing an allocation and
testing it.**

The first experiment guessed two allocations and both failed. Worse, it spent an
arm on the assumption that linear-attention could absorb a cut — and the data
said the opposite. A sensitivity sweep produces that answer directly, once, for
every tensor group, and every future allocation follows from it.

## Method

One-group-at-a-time perturbation against a near-lossless baseline.

1. Baseline: **everything Q6_K** (~22 GB, near-lossless; measured KL vs BF16
   gives the floor).
2. For each group G: take the baseline and drop **only G** to **Q3_K**.
   Everything else stays Q6_K.
3. Sensitivity of G = ΔKL vs the baseline.

Q3_K is a deliberately large perturbation. The first experiment failed partly
because it probed a near-lossless band where nothing separated; here we want a
big, clearly-resolvable signal per group.

The number that decides allocation is not ΔKL but **ΔKL per GB saved** — the
exchange rate. Bits should be taken from whichever group is cheapest per byte,
which is exactly the quantity nobody has measured on this architecture.

## Arms

Groups follow the parameter budget (27.3 B, excl. vision):

| # | group | tensors | share |
|---|---|---|---|
| 1 | ffn_gate | `blk.N.ffn_gate.weight` | 20.9% |
| 2 | ffn_up | `blk.N.ffn_up.weight` | 20.9% |
| 3 | ffn_down | `blk.N.ffn_down.weight` | 20.9% |
| 4 | linattn_qkv | `blk.N.attn_qkv.weight` | 9.1% |
| 5 | linattn_gate | `blk.N.attn_gate.weight` | 5.4% |
| 6 | linattn_out | `blk.N.ssm_out.weight` | 5.4% |
| 7 | fullattn | `blk.N.attn_q/k/v/output.weight` | 6.1% |
| 8 | token_embd | `token_embd.weight` | 4.7% |
| 9 | output | `output.weight` | 4.7% |

Plus two arms that test the layer-position hypothesis directly, at high signal —
this is the question the first experiment could not resolve:

| # | arm | what it isolates |
|---|---|---|
| 10 | ffn_mid_only | drop FFN to Q3_K in middle layers only (10–53) |
| 11 | ffn_edge_only | drop FFN to Q3_K in edge layers only (0–4, 59–63) |

If edge and middle come back with the same ΔKL **per GB**, the layer-position
gradient is dead on this architecture, at any band — a much stronger statement
than the first experiment could make.

Total: 1 baseline + 11 arms = **12 quants**.

## Cost

Cheap, because the expensive inputs already exist on thor:

- `imatrix.gguf` — reuse (all arms must share one imatrix anyway)
- `reference-logits.bin` (25.3 GB) — reuse, so KL needs no new BF16 pass
- BF16 — already staged, and archived on the NAS

Per arm: ~5 min quantize + ~12 min PPL+KL (measured last run) ≈ 17 min.
12 arms ≈ **3.5 h**, one flock window on thor.

## Execution rules (all learned the hard way)

- **One `flock /tmp/gpu` for the whole sweep**, quantize included — thor has
  unified memory, so memory-heavy CPU work contends with GPU work.
- **Quantize in a memory-capped container** (`--memory=40g`), so an OOM kills the
  container and not the box.
- **Verify each quant's size against prediction** (±3%) before measuring; a
  truncated file is non-empty and will otherwise sail through.
- Jetson/Tegra: `--runtime nvidia -e NVIDIA_DISABLE_REQUIRE=1`, and put
  `/opt/nvidia/l4t-gpu-libs/nvgpu` on `LD_LIBRARY_PATH`. `--gpus all` does not
  work there.
- One imatrix, one reference-logits file, shared by every arm.

## Decision rule — fix this before looking at the numbers

- If the ΔKL-per-GB spread across groups is **small** (say within ~2x), then
  precision is roughly fungible across this architecture, the shelf's flat
  allocation is near-optimal, and **dense is closed**. Write it up and stop.
- If some group is **markedly cheaper per GB** than the rest, build one
  allocation that takes bits from it and gives them to the most expensive
  group, and test that at the **Compact/Mini band (~12–14 GB)** — where the
  first experiment suggests the headroom actually is.
- Report the sensitivity table either way. A measured sensitivity curve for a
  hybrid-attention dense model is a useful artifact even if it kills the idea,
  and nobody else appears to have published one.

## What this does not test

Speed. Everything here is quality-per-byte. Throughput effects of the resulting
allocations (cache behaviour, kernel selection per quant type) are a separate
question and would need the box quiet and a proper A/B under one lock.
