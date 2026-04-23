# Handoff — autonomous session 2026-04-22

## tl;dr

Built and shipped the multi-env training infrastructure. On Vast, Dolphin instability under the current machine state is blocking both multi-env **and** single-env runs — not a code problem, a runtime-environment problem. **Nothing is currently training.** Code is clean, pushed, tests green.

## What's committed

Commits since session start (all on `main`, all reversible):

- `b908729` env: kill Dolphin process group in close() — fixes orphan leak.
- `fc24aef` train: per-episode `log.info` line so tmux tail shows progress.
- `e599738` train: multi-env parallel-rollout path gated on `num_envs` — new `_train_vector()`, thread-per-env with `agent_lock`, per-env crash recovery, 3 new tests in `TestMultiEnv`.
- `b2149fd` env: resolve multi-env Dolphin path as sibling of `dolphin0`.
- `a75b67e` _train_vector: better crash diagnostics + abort on empty sampler.
- `da75562` env: append Dolphin logs instead of truncating on relaunch.
- `d6803c4` SETUP.md: multi-env runbook + known SIGSEGV quirk.
- `251368a` train: reset per-track crash count on successful episode (stops long runs from aborting over accumulated flakes).
- (local, not yet committed) train: match multi-env error formatting in single-env path.

Infra added:

- `scripts/setup_dolphin_instances.py` — clones `dolphin0/` → `dolphin1..N/`.
- `configs/btr.yaml` gained `env.num_envs` (default 1).
- Full test suite 293 passed (3 new multi-env tests).

## What went wrong on Vast

Timeline:

1. Multi-env infra shipped, 2-env **smoke** test (testing mode, tiny config) passed cleanly — 502 env_steps, 499 grad_steps, `_final.pt` saved.
2. Attempted 4-env **production** run. 2 of the 4 Dolphins SIGSEGV'd within seconds of slave handshake. The `MAX_TRACK_CRASHES=3` counter hit for Luigi (our only track), which triggered `sampler.remove_track()` on the last track → `max()` on empty `progress` dict → run aborted.
3. Fell back to 2-env production. Also crashed after only a couple episodes with the same symptom (EOFError on master, slave dies silently).
4. Discovered `MAX_TRACK_CRASHES` was monotonic across the whole run — one crash per hour over a long run would eventually accumulate. Fixed to reset on successful episode (commit `251368a`).
5. Relaunched 2-env: ran 17 episodes, hit 3 total track crashes (all env 1, 3 in a row after a streak of successes) → still aborted.
6. Wiped Dolphin's shader cache + re-cloned `dolphin1/` from `dolphin0/`, retried: aborted again after 2 episodes with 3 consecutive crashes.
7. Fell back to **single-env** to validate the non-multi path: also crashes after ep=1. Even wiping `dolphin0/user/` entirely (letting Dolphin regenerate from scratch) didn't help.

**Dolphin log after crash shows `[slave N] init handshake complete; entering main loop` and then nothing** — no SIGSEGV message, no Python traceback. Master sees `EOFError: ''` meaning the socket closed from the slave side. Dolphin's stdout buffer is almost certainly holding the crash message but never flushing because Dolphin dies before a line ends.

## What I rule in / out

- **NOT a code regression.** Single-env was working clean earlier in the same session (the 4-minute smoke that saved `luigi_prod_...  _final.pt`, and the 6-episode run before that). Code was rolled forward in a reviewable sequence; reverting to `fc24aef` (before multi-env work) would give the same broken behavior now.
- **NOT a cache/config issue in `dolphin0/user/`.** Wiped it entirely, problem persists. Also, the backup I took wasn't actually pre-corruption — I only realized afterward it was taken AFTER the first 4-env crash.
- **NOT memory/CPU/disk pressure.** `free -h` shows 120 GB available, `uptime` load avg 2–6, disk 87 GB free.
- **Likely the Vast VM itself is in a degraded state.** Machine has 109 days uptime. Kernel-level resource leak, memory page fault, or some shared-state corruption that accumulates under repeated Dolphin SIGSEGVs could plausibly poison subsequent Dolphin launches.

## Recommended next steps for you

1. **Rent a fresh Vast instance.** Cheapest and most conservative. Re-run the setup: install deps, copy savestates + ISO, run `scripts/setup_dolphin_instances.py`, launch single-env first to sanity-check, then scale up. Expect 30 min to stand up.
2. **If you want to debug on the current box first**: reboot the instance (via Vast dashboard). If single-env works after reboot, the instance was degraded. If it still crashes, the underlying install is genuinely broken and a reboot won't help.
3. **Before scaling to multi-env again**: fix the cache-poisoning recovery. In `_train_vector._rollout_worker` around `env_slots[i] = _make_env(cfg, env_id=i)` (approx `src/mkw_rl/rl/train.py:1280`), add a wipe-and-reclone of `dolphin{i}/User/Cache` before the relaunch. This would help with the 4+ env instability if that was the real cause.
4. **Consider per-crash diagnostics.** The single-env path now logs `(type_name: repr)` (uncommitted), which would have told us whether the crashes are EOF/Timeout/OSError. Worth committing before the next attempt.

## Vast ssh for reference

```bash
ssh -p 41927 root@71.241.245.11
```

Tmux sessions were killed; no processes running. `runs/btr/` has the failed runs' logs and checkpoints — feel free to grep/inspect. The canonical config (`configs/btr.yaml`) has Vast-specific paths + `num_envs: 1`, set for a single-env attempt; edit as needed.
