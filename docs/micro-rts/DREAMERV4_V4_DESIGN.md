# DreamerV4 v4 — Architecture Intuition, Objective Design, and the Experiment Plan

*2026-07-10. Status: v4 implemented and smoke-verified; tokenizer_v4 and dynamics_v4
runs not yet launched. Companion history: `NEXT_PLAN.md` (v3 joint-action plan),
`docs/micro-rts/NOTEBOOK.md`.*

## 0. Where this is going

The research target is **multi-agent, incomplete-information world modeling**:
agents that plan inside a learned model of an environment that contains *other
agents whose actions (and eventually observations) they cannot see*. MicroRTS
is close to a perfect substrate for this — two-player, zero-sum, spatially
structured, cheap to simulate at scale, with an engine we control (patched jar
exposes both players' actions and pre-reset terminal frames), and a natural
fog-of-war extension for true partial observability later.

Everything below is in service of a staged plan:

1. **Get the recipe right** (now): a world model whose forward predictions are
   *causally conditioned* on both players' actions, verified by counterfactual
   probes — on a small corpus, before any scale.
2. **Probe and optimize inference** (next): once the dynamics model passes its
   gates, profile the forward pass and imagination rollout for inefficiencies.
   KV-caching is deliberately out of scope while eval is discrete/offline;
   kernel-level wins (flash attention, bf16, TF32, fused optimizers) are in
   scope now because they're free and don't change the architecture.
3. **Scale + the actual research** (after): bigger corpus, fog-of-war variants,
   opponent modeling inside imagination (`imagine(opponent_policy=...)` is
   already plumbed), league-style asymmetric-information experiments.

The failure mode this document exists to prevent: a model that *looks* good on
aggregate losses while being useless for planning. That exact thing happened
twice (v2: expert-only data made actions carry no information; v3: uniform
spatial averaging made the action signal invisible to the gradient). The
recipe is as much about **measurement** as about architecture.

## 1. The architecture, and why each piece

The stack is Dreamer 4 (Hafner et al. 2025) adapted from pixels to a
categorical grid, trained in three separate optimization problems:

```
obs (27,H,W one-hot planes)                    action (H*W cells x 7 comps) x 2 players
        │                                                    │
   GridTokenizer (CNN, /4)                        GridActionEncoder (CNN, /4)
        │  continuous latents                                │  action tokens
        ▼  (H/4*W/4 cells x 32-d, tanh)                      ▼
     [frozen after phase 1]              ┌──────────────────────────────────┐
        z ──────────────────────────────▶│  block-causal transformer (47.6M) │
                                         │  space attn (full mixing) +       │
                                         │  causal time attn, shortcut-      │
                                         │  forcing flow denoiser            │
                                         └──────┬─────────────┬─────────────┘
                                                ▼             ▼
                                          x1_hat (denoised   registers →
                                           next latents)     reward (symlog two-hot,
                                                             DENSE) + continue heads
```

**Spatial continuous latents, not vectors, not VQ.** The board is a grid; unit
interactions are local; the policy head (GridNet) is per-cell. A spatial latent
keeps that inductive bias end-to-end. Continuous (tanh) rather than discrete VQ
tokens because the dynamics objective is Dreamer 4's *shortcut-forcing flow*
(x-prediction with diffusion forcing + dyadic self-consistency), which needs a
continuous space to denoise in. Measurement backs this up — the latent-delta
probe (§3) shows the v3 tokenizer's latent geometry is already sharply
discriminative, so the representation family is not the problem.

**Joint-action conditioning (v3).** Both players' per-cell actions are embedded
with *separate* embedding tables (a shared table summed with a role embedding is
order-invariant — it cannot represent who did what), plus a learned
`unknown_opp` embedding with `opp_dropout: 0.15` so one model serves both the
conditional (opponent visible) and marginal (opponent unknown) dynamics. This
is the incomplete-information hook: at RL time, imagination can either
marginalize over the opponent or be driven by an explicit `opponent_policy`.

**Dense reward head, symlog two-hot.** The reward head is trained on the shaped
dense reward (`reward_weight: [10, 1, 1, 0.2, 1, 4]` — win, resources, units,
etc.), not just terminal win/loss, using DreamerV3's symlog two-hot
classification (255 bins). This matters enormously for the RL phase: the reward
head is the *only* learning signal the imagined actor gets, and a dense,
well-calibrated one gives gradient at every imagined step. v3 already passes
this gate (corr 0.86 on nonzero-reward transitions; terminal sign accuracy
0.996). **Do not change it.**

**Frozen-tokenizer staging.** Phase 1 trains the autoencoder alone; phase 2
trains the dynamics on frozen latents (unit-RMS normalized via the measured
`latent_scale`). Upstream-faithful, and it makes failures attributable: any
dynamics problem is a dynamics problem, not a moving-representation problem.

## 2. What v3 got right, and exactly how it failed

The v3 eval (`eval_dreamer_dynamics.py`, checkpoint step 60k) was **NO-GO**
with a very specific signature:

| Gate | Result | Meaning |
|---|---|---|
| reward head | **PASS** (corr .86) | knows how good a state is |
| continue head | **PASS** (AUC .98) | knows when episodes end |
| imagination stability | **PASS** (no drift/NaN) | rollouts don't explode |
| action conditioning | **FAIL** (CF ratio 1.02) | *doesn't know actions matter* |
| opponent conditioning | WARN | same, opponent channel |
| beats frozen copy-last | WARN | dynamics add ~nothing over "board stays put" |

So the model is an excellent *state evaluator* and a non-functional *dynamics
simulator*. The training-time counterfactual probe (CF-PROBE, logged every 1k
steps) shows `self_gap` — open-loop MSE with shuffled actions minus with true
actions — oscillating around **zero for all 60k steps**. The model never began
to use the action channel.

**Root cause is gradient signal-to-noise, not data and not representation:**

- The v3 corpus is fine: verified format v3, 1.44M steps, ε-noised/self-play/
  random mix breaks the state→action determinism that killed v2.
- The tokenizer is fine: 97–99% occupied-cell reconstruction, and the
  latent-delta probe (§3) shows 24x separation between changed and unchanged
  cells.
- But per frame, only ~1.1% of cells change (`obs/changed_cell_frac = 0.011`).
  The flow-matching MSE averages **uniformly over the spatial latent grid**, so
  the loss (and its gradient) is ~99% "reproduce the static background" and ~1%
  "predict what the actions did". A 47.6M transformer minimizes that objective
  the obvious way: nail the background, ignore the actions. Aggregate loss goes
  down; the CF probe stays at zero; the reward head still works because reward
  correlates with board *state*, which the model tracks fine.

Framed as the question asked during diagnosis: it is a **signal-to-noise
problem, not a signal-amount problem**. The action-effect signal is present in
every batch (the corpus records both players' actions and the board deltas);
it's just diluted 100:1 at the loss. More data at the same SNR would train the
same blind model more thoroughly. That's why the v4 recipe changes the
*objective*, holds the data fixed, and only then considers scale.

## 3. Measurement before surgery: the latent-delta probe

Before touching the tokenizer, we measured whether it was actually the weak
link. In the unit-RMS latent space the dynamics model operates in (per latent
cell, v3 tokenizer, v3 corpus):

| cells | latent delta RMS t→t+1 |
|---|---|
| changed (board state differs) | **0.62** (p10 0.40) |
| unchanged | **0.026** (p90 0.076) |
| ratio | **24x** |

Interpretation: the encoder linearly separates "something happened here" from
background by more than an order of magnitude, with clean tails. The latent is
*good*. Two consequences:

1. No autoencoder replacement (VQ, vector bottleneck, etc.) is justified —
   swap-the-architecture instincts would have burned a week for nothing.
2. The remaining tokenizer-side opportunity is quantified: changed cells carry
   ~0.6 RMS of signal against the flow objective's unit-variance noise. If a
   sharper recon objective pushes that up, the dynamics model's denoising
   target gets easier. That's a *refinement*, not a fix.

This probe should be rerun after tokenizer_v4 trains (gate: changed-cell delta
≥ 0.62, occupied-cell accuracy ≥ v3's).

## 4. The v4 objectives

Both changes are default-off; all-1.0/0.0 config values reproduce v3
behavior byte-for-byte, so every run is a clean ablation.

### 4.1 Dynamics: cell-weighted flow matching (`loss/dreamer.py::cell_weights`)

Per-spatial-cell weights on the flow MSE (both the empirical term and the
dyadic self-consistency term), computed from the raw obs and max-pooled to the
tokenizer's /4 latent grid:

- static background cells: `floor = 1.0`
- occupied cells (any unit): `occ_boost = 4.0`
- cells whose one-hot state changed vs the previous frame: `changed_boost = 16.0`
- combined by **max**, not sum; **mean-normalized to 1 per frame** so the
  overall loss scale (and `flow_coef`) is untouched — only the intra-frame
  *distribution* of gradient moves.

This is the standard remedy for extreme foreground/background imbalance in
dense prediction (focal-loss-style reweighting; occupancy-only losses in
voxel/video world models). With ~10% of latent cells changed per step and a
16x boost, changed cells go from ~10% to ~2/3 of the gradient mass.

### 4.2 Tokenizer: group cross-entropy + the same cell weighting

Two additions to `tokenizer_loss`, motivated by what works in grid-game world
models (IRIS trains its dynamics on discrete tokens with categorical CE —
much of its sample efficiency is attributed to proper categorical likelihoods;
Delta-IRIS focuses its autoencoder on inter-frame *deltas* and holds SOTA on
Crafter):

- **`group_ce_coef = 0.3`** — the 27 obs channels are five one-hot groups
  (hp/resources/owner/unit_type/action); within a group, the decoder output is
  a categorical distribution and cross-entropy is its proper likelihood. CE
  puts real gradient on rare classes (units) that per-channel MSE
  underweights. The MSE term stays: together with `latent_noise: 0.1` it
  anchors the tanh-latent scale (the run-1 latent-collapse lesson).
- **`cell_occ_boost 4 / cell_changed_boost 16`** at `downsample=1` (recon lives
  on the raw grid, unlike the flow loss) — encoder capacity flows to dynamic
  content.

### 4.3 What was deliberately NOT changed

- **Architecture** — transformer, tokenizer, action encoder all byte-identical
  to v3. One variable at a time.
- **Reward/continue heads** — passing their gates; dense reward is load-bearing
  for RL.
- **The corpus** — `tokdyn_pretrain_v3` validated solid (schema, invariants,
  provenance mix, obs one-hot checks). Scale comes after the recipe works.
- **Inverse-dynamics auxiliary head** — a candidate escalation (predict a_t
  from z_t, z_{t+1}; directly forces action information into the latent) if
  cell weighting alone doesn't move the CF probe. Not added now: the E-gates
  measure action causality directly, and fewer moving parts wins.

## 5. Experiment plan

Two independent runs, launchable in parallel (they share nothing but the data):

```bash
# A. dynamics_v4: cell-weighted flow, FROZEN v3 tokenizer  (the primary fix)
python src/micro-rts/entrypoints/train_dreamer_dynamics.py \
    --exp micro-rts/pretrain_dreamerv4_dynamics_v4

# B. tokenizer_v4: group-CE + cell-weighted recon           (the refinement)
python src/micro-rts/entrypoints/train_dreamer_tokenizer.py \
    --exp micro-rts/pretrain_dreamerv4_tokenizer_v4
```

Decision tree:

1. **Watch dynamics_v4's CF-PROBE from step ~1k.** The go/no-go signal is
   `probe/self_gap` (and `opp_gap`) trending **positive and growing** — not the
   aggregate flow loss. If still flat at ~10k steps, stop the run; retune
   boosts (e.g. changed 32–64x / floor 0.5) or escalate to the
   inverse-dynamics head. Don't spend the full 60k on a flat probe.
2. **If dynamics_v4 passes** (`eval_dreamer_dynamics.py` action_conditioning
   PASS, open-loop beats frozen copy-last): the flow weighting was the fix.
   Then evaluate tokenizer_v4 (occupied-cell acc + latent-delta probe) and, if
   it improves on v3, retrain dynamics once more on top of it
   (`--tokenizer-ckpt checkpoints/dreamer_tokenizer_v4.pt`) for the best
   pairing. That checkpoint is the RL candidate.
3. **If dynamics_v4 fails but tokenizer_v4's latents are sharper**: retrain
   dynamics on tokenizer_v4 with cell weighting — the combination attacks the
   SNR from both sides.
4. **Only after a PASS**: RL (`rl_dreamerv4_hybrid`), and only then corpus
   scale-up. Scaling a NO-GO recipe multiplies the waste.

Gates (unchanged from NEXT_PLAN.md, they caught both previous failures):

| gate | threshold |
|---|---|
| CF probe, training | self/opp gap > 0 and growing with horizon |
| action_conditioning, eval | counterfactual/real step-1 MSE ratio > 1 (FAIL at ≤1) |
| open loop vs frozen copy-last | occupied-cell acc delta > 0 over first 5 steps |
| reward head | corr ≥ v3's 0.86 (must not regress) |
| continue head | AUC ≥ v3's 0.98 (must not regress) |

## 6. Throughput work done now (no architecture change)

Attention was already `F.scaled_dot_product_attention` with the all-True
`wm_agent` space mask dropped to `None` (explicit masks disqualify the flash
kernel) and `is_causal=True` time attention — i.e. flash-*eligible*. But the
pretrain loops ran **fp32, where SDPA cannot dispatch to FlashAttention at
all**. Fixed in `pretrain_common.py`, verified on the real 47.6M model:

- **bf16 autocast** (`training.amp: true` in both v4 configs) — SDPA now takes
  the actual flash kernel (verified via `torch.nn.attention.sdpa_kernel`:
  flash OK under bf16, RuntimeError under fp32). `cache_enabled=False`
  mirrors the RL trainer's fix for the no-grad/grad weight-cache pitfall.
- **TF32 matmuls + cuDNN autotune** (`setup_backend`) — covers the conv-heavy
  tokenizer/action-encoder trunks and any fp32 residue.
- **Fused Adam** (`make_adam`) — one multi-tensor CUDA kernel per step.

Measured, full training step (fwd+bwd, B16xT16, real checkpoint):
fp32 461 ms → TF32 385 ms → **bf16+TF32 343 ms (1.34x)**, loss/grads verified
finite. The remaining time is dominated by the CNN action encoder and
per-frame conv work, not attention — which is exactly what to profile in the
"inference inefficiencies" pass after the recipe converges (candidates, in
rough order: batching the two action-channel CNN passes into one, `torch.compile`
on the denoiser, and only-then anything attention-shaped).

## 7. Recipe principles (the meta-lessons so far)

1. **Aggregate losses lie in imbalanced domains.** Every failure so far hid
   behind a healthy-looking loss curve. The probes (CF action shuffle,
   latent-delta, occupied-cell accuracy, copy-last baselines) are the actual
   instrument panel; build them before training, watch them during.
2. **Diagnose with measurements, not plausible stories.** "The tokenizer might
   be bad" was a reasonable hypothesis; a 20-line probe falsified it in
   minutes and redirected the fix to the objective.
3. **One variable per run, defaults that reproduce the previous version
   exactly.** v4 configs differ from v3 only in the new fields; ablation comes
   free.
4. **Fail fast by gate, not by step budget.** A flat CF probe at 10k steps
   means stop, not hope.
5. **Fix SNR before adding scale.** Data volume multiplies whatever gradient
   signal exists — including zero.

## References

- Hafner et al., *Training Agents Inside of Scalable World Models* (Dreamer 4), 2025 — https://arxiv.org/abs/2509.24527
- Micheli et al., *Transformers are Sample-Efficient World Models* (IRIS), 2022 — https://arxiv.org/abs/2209.00588
- Micheli et al., *Efficient World Models with Context-Aware Tokenization* (Delta-IRIS), ICML 2024 — https://arxiv.org/abs/2406.19320
- Lin et al., *Focal Loss for Dense Object Detection*, 2017 — https://arxiv.org/abs/1708.02002 (the foreground/background reweighting lineage)
- Hafner et al., *Mastering Diverse Domains through World Models* (DreamerV3; symlog two-hot, percentile return normalization), 2023 — https://arxiv.org/abs/2301.04104
