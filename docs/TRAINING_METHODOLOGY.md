# Training methodology

Concrete design decisions for our track-agnostic multi-track BTR training. Each decision is attributed to the source that justified it.

## Source material

Two YouTube videos by AI-Tango (the author of VIPTankz/Wii-RL and the BTR paper):

- **v1 (3-week attempt, 19/32 tracks incomplete)**: `youtube.com/watch?v=6OofM-Q3dGA`. Initial multi-track attempt. Uses their published `BTR.py` architecture (IMPALA + IQN + Noisy + Dueling + PER, frame-stack only, no recurrence). Fails on Rainbow Road and Bowser's Castle variants.
- **v2 (500-hour attempt, podium on all 32 tracks)**: `youtube.com/watch?v=aLw43abG-NA` — the breakthrough run. Same BTR core but with five specific changes that collectively made the policy generalize to every track.

**Key context for our project**: the v2 architecture and training-loop changes are **not in VIPTankz's published `BTR.py`**, which corresponds to v1 / single-track. Forking their code verbatim gives us the inferior algorithm. We implement the v2 changes on top.

## Five load-bearing design choices

### 1. Variable checkpoints per track, scaled by world-record time

**What**: each track gets `round(100 × WR_time_minutes)` checkpoints per lap, where `WR_time_minutes` is the no-glitch world-record time for that track. Not 200 for every track.

**Why** (v2 video, ~3:00): v1 used a fixed 200 checkpoints per lap on every track. Short tracks like Luigi Circuit got dense reward signal; long tracks like GBA Bowser's Castle 3 got the same count spread over far more geometry, so the reward signal was much sparser on long tracks. Sparser reward → harder learning → v1's observed failure on long tracks.

**Implementation hook**:
- `data/track_metadata.yaml` (new) — lookup table keyed by track slug, with `wr_seconds_no_glitch` for each of the 32 vanilla tracks. Source values from `speedrun.com/mkwii` leaderboards, no-glitch category. User curates this file.
- `src/mkw_rl/env/reward.py` (new) — `checkpoint_count_for_track(slug)` function returns `round(100 * wr_seconds / 60)`. Checkpoint placement itself is whatever scheme VIPTankz's `DolphinScript.py` uses (course-geometry-based); only the count changes.

### 2. LSTM on top of IMPALA encoder — not frame-stack-only BTR

**What**: architecture is `(B, T, 4, 75, 140) → IMPALA CNN per-timestep → LSTM (hidden=512, 1 layer) → IQN dueling heads → Q-values over 40 discrete actions`. The LSTM is between the encoder flatten and the Q-heads.

**Why** (v2 video, ~5:16-6:15): v1 had no memory. When the policy got into a novel state (stuck, facing backward, bullet bill active, item just used), the 4-frame stack gave no context about how it got there. LSTM lets the policy remember the last N-step trajectory and recover. v2 author called this "the biggest of all changes."

**Implementation notes**:
- Our existing `src/mkw_rl/bc/model.py` already implements exactly this encoder+LSTM combination for BC. Reuse the `ImpalaEncoder` and LSTM module verbatim; swap the discrete-action heads for IQN heads from VIPTankz's `BTR.py`.
- Recurrent DQN with PER is known territory (R2D2 pattern). Replay buffer must either (a) store burn-in sequences long enough for the LSTM to warm up, or (b) store saved LSTM states at trajectory boundaries. We default to (a) with a burn-in length of ~20 frames unless profiling shows otherwise.
- **Action-space note**: BC uses 21-bin steering + 4 binary button heads (a supervised-learning choice for matching `.dtm` controller states); BTR uses `Discrete(40)` (VIPTankz's pre-enumerated action set in `DolphinEnv.py:94`). These are independent design choices — the warm-start transfers **only the encoder+LSTM weights**, not the heads. No attempt is made to map the 21-bin steering space into the 40-action space.
- Consequence for BC-augmentation (future Phase): because BTR shares encoder+LSTM with BC, the warm-start path is clean — load BC's encoder+LSTM weights, leave the IQN heads randomly initialized (or pretrain them separately).

### 3. Lenient reset threshold — progress-based, not crash-based

**What**: episode resets only when the agent fails to make ~1 second of forward progress within a 15-second window. **Never** resets on falling off an edge, hitting a wall, or similar events. Those are handled via reward penalty (see #5).

**Why** (v2 video, ~1:10-2:30): v1 reset on any off-edge event. This unfairly biased training against tracks with more edges (Rainbow Road, Bowser's Castle), since an item hit or minor mistake would end the episode immediately and the policy got too few samples of those tracks. Lenient reset + harsh reward penalty gives the same behavioral shaping without the sample-efficiency penalty.

**Implementation hook**:
- `src/mkw_rl/env/dolphin_env.py` (new) — track progress via last-checkpoint-hit-time. If `current_time - last_checkpoint_time > 15s` AND the net forward progress in that window is < 1s worth of normal driving, issue `done = True` and reset.
- Edge-fall and wall-contact events become reward modifiers (see #5), not termination signals.

### 4. Progress-weighted track sampler, not uniform random

**What**: after `reset()`, the next track to load is chosen by a weighted formula that biases toward tracks where the agent is making the least progress (not just the least time spent).

**Why** (v2 video, ~4:10-5:00): v1 sampled tracks by "least time played." That sounds balanced but creates a vicious cycle: if the agent is bad at track X, it keeps resetting quickly there, accumulating less wall-clock time on X, so it gets sent back to X often — but the short episodes give too little learning signal per attempt. Better signal comes from weighting on _progress per attempt_: force the agent to stay on tracks where its episode reward is plateaued-low.

**Implementation hook**:
- `src/mkw_rl/rl/track_sampler.py` (new) — maintains an EMA of episode reward per track. Sampling weight per track ∝ `(max_progress_across_tracks - track_progress + epsilon)`. After the agent "solves" a track (progress ≥ some fraction of best), its weight drops to baseline and the sampler redistributes to harder tracks.
- Log the current per-track sampling distribution as wandb metric `track_sampler/{slug}/weight` so we can watch the curriculum evolve.

### 5. Reward function: variable checkpoints + finish bonus + off-road/wall penalty

**What**, additive per frame:

| Component | Value | Notes |
|---|---|---|
| Checkpoint-hit reward | `base × speed_bonus` | `base = checkpoint_reward_per_lap / N_checkpoints_for_track` (config default: `1.0`), `speed_bonus` = multiplier rewarding fast checkpoint hits. Normalizes so total per-lap checkpoint reward is ~constant across tracks. |
| Off-road penalty | `-c_offroad` per frame while off-road | Small. Tune so that intentional shortcuts through off-road (with a mushroom) are still net positive. |
| Wall-contact penalty | `-c_wall` per frame on wall | Larger. |
| Finish bonus | `+R_finish` on race completion | Large. |
| Position bonus | `+R_pos × (n_racers - finishing_pos)` | Much smaller than `R_finish`. |
| Progress-based shaping (VIPTankz's existing) | retained | Keeps dense signal between checkpoints. |

**Why** (v1 video ~1:35 for checkpoint/speed shape; v2 video ~9:00-10:20 for penalty structure): v1 made off-edge a hard reset; v2 converts all of "agent did a bad thing" into reward-level signals so the agent can recover and keep learning in-episode. The "small penalty for off-road" on Shy Guy Beach even led the v1 AI to _learn_ that mushrooms eliminate the penalty when going over water — an emergent strategy. Keep that emergent-learning property.

**Implementation hook**:
- `src/mkw_rl/env/reward.py::compute_reward(state, prev_state, track_meta)` returns a scalar per frame.
- Reward component breakdown logged separately to wandb (`reward/checkpoint`, `reward/offroad`, `reward/finish`, `reward/position`) so we can diagnose shaping issues.

## Other v2 observations worth preserving

These are not primary design levers but context we want on hand during training review:

- **Item-use policy is hard to learn**. v2 at the end was still confused by item chaos. Expect our policy to look "sloppy" even when scoring well; this is normal.
- **Backward driving can be rewarded**. If off-road/wall penalties are tuned too punitive relative to checkpoint signal, the policy sometimes prefers driving backward (avoiding walls) over hitting checkpoints. Sanity-check by watching episode video at 150M+ steps.
- **Bullet Bill is over-used early**. v2 policy tried to activate bullet bill mid-spin-up, wasting it. Not a design problem; just noise.
- **Blooper is the hardest single item for vision-based policy.** Screen obscurement breaks the image encoder's state representation in ways the human player doesn't experience. Consider logging per-blooper-event rewards to quantify the hit.
- **Gap-jump shortcuts are emergent**. v1's policy attempted Mushroom Gorge's Gap Jump unprompted. Take this as signal that reward shaping is not destroying exploration.
- **Plateau timing**: v2 trained 500 hours / ~700M frames and was still improving at the end — likely room for much longer runs. For us, target first checkpoint review at ~5M env steps (early signal), second at ~50M (meaningful policy quality), third at ~500M (approaching v2 scale).
- **Hard-track sample share**: v2 spent ~80% of late training on Rainbow Road alone due to the progress-weighted sampler. Budget for this — it's not a bug.

## Open questions / deferred

- **Exact speed_bonus curve**. v2 video shows it's "reward scales with speed of hitting checkpoint" but doesn't publish the formula. Start with linear (`1.0 + α × (1 - elapsed_since_last_checkpoint / expected_gap)`, clamped) and iterate.
- **Per-track reward normalization**. If some tracks end up dominating gradient even after variable checkpoints, consider z-scoring per-track returns before feeding into BTR's loss.
- **LSTM state at replay time**. Start with burn-in approach. If memory footprint is a problem, fall back to stored-hidden-state approach (R2D2 style).
- **World record times to populate `data/track_metadata.yaml`.** User task. No-glitch category on speedrun.com/mkwii.
- **Retro Rewind tracks reward function**. Out of scope (region conflict anyway — see `docs/REGION_DECISION.md`).

## Inherited implementation quirks (2026-04-21 VIPTankz audit)

A forensic audit of VIPTankz/Wii-RL's BTR.py turned up several deviations from the PER paper (Schaul et al. 2016) and the IQN paper (Dabney et al. 2018) that VIPTankz's code carries. These are **deliberate** in our port — we match VIPTankz's code so our numbers stay directly comparable to their published BTR results. Once training is stable on multi-track, we plan to A/B test "faithful PER" vs "VIPTankz PER" on Luigi Circuit to see whether the accidental divergences actually help or hurt.

PER deviations (documented in `src/mkw_rl/rl/replay.py` module docstring as "Known Deviations"):

1. **PER importance-sampling exponent uses `alpha` instead of `beta`** (replay Known Deviation #1). VIPTankz's comment at `BTR.py:622` acknowledges this was accidental ("performs better"). Net effect: IS correction is weaker than spec; training sits closer to vanilla prioritized sampling.
2. **No PER beta annealing** (replay Known Deviation #2). `priority_weight_increase` exists in VIPTankz's Agent but is never consumed in the sampler (since the sampler uses alpha per #1). Our port drops it entirely.
3. **Batch-min IS weight normalization** (replay Known Deviation #3) rather than buffer-min. `weights / weights.max()` where the theoretically-correct normalizer is the min priority across the whole buffer.
4. **Raw |δ| priority** (replay Known Deviation #4) instead of quantile-Huber-adjusted. Dopamine and ku2482 use the Huber loss value; VIPTankz uses raw TD magnitude.
5. **Storage multiplier bumped 1.25→1.75** (replay Known Deviation #5; this one is our own MKWii tuning, not a VIPTankz deviation). VIPTankz's 1.25 sizes the frame pool for Atari's ~20-frame episodes; MKWii episodes are ~1000 frames, so 1.25 is below their own back-of-envelope minimum. Configurable via PER constructor.

Architecture deviation (documented inline in `src/mkw_rl/rl/model.py`):

6. **Cosine embedding index `{0..n_cos-1}`** (VIPTankz convention) instead of the canonical `{1..n_cos}` from Dabney eq. 4. The `i=0` term gives `cos(0)=1` for every τ — wastes one basis dim on a constant the Linear's bias already provides. Kept for weight-transfer compatibility with VIPTankz's pre-trained model.

Training-loop deviations (documented in `src/mkw_rl/rl/train.py` module docstring):

7. **Quantile-Huber axis convention per Dabney eq. 10** — sum over target-τ, mean over online-τ. VIPTankz's BTR.py:1076 has these swapped. Mathematically identical when `num_tau_online == num_tau_target` (our default), but the Dabney form is correct if we ever decouple the two tau counts.
8. **Priority signal uses `mean(dim=tau).mean(dim=tau)`** (scale-invariant) rather than VIPTankz's `sum(dim=tau).mean(dim=tau)` which scales with num_tau. Avoids priorities drifting with num_tau changes during tuning.
9. **Sequence-level priority aggregation** per R2D2 §2.3 eq. 1: `η·max_t|δ_t| + (1-η)·mean_t|δ_t|` with `η=0.9`. VIPTankz is feed-forward so they update transition-level priorities directly; our LSTM variant needs this aggregation.
10. **Target-net sync cadence 200 grad steps** (not VIPTankz's 500). MKWii's non-stationarity across tracks rewards a faster-tracking target. Revisit if A/B testing shows it doesn't matter.
11. **No ε-greedy schedule** — we drop VIPTankz's 100M-frame ε-disable schedule (`BTR.py:943`) and rely on noisy-nets exploration alone. Simplifies the training loop; if exploration is too weak early, reintroduce.

See `src/mkw_rl/rl/replay.py`, `src/mkw_rl/rl/model.py`, and `src/mkw_rl/rl/train.py` docstrings for full per-item analysis.
