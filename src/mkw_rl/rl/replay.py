"""Prioritized Experience Replay + SumTree — ported verbatim from VIPTankz's ``BTR.py``.

Two classes:

- ``SumTree``: binary-indexed tree over leaf priorities; O(log n) sampling
  proportional to priority and O(log n) updates. ``find(values)`` returns
  leaf tree indices whose cumulative priorities sum to the given targets
  (inverse CDF via tree traversal).
- ``PER``: prioritized replay buffer with frame-stacking and n-step returns.
  Stores raw frames once and indexes into them via a pointer table —
  significantly reduces memory vs storing stacked frames per transition.
  Handles multi-env rollouts via ``stream`` IDs (pre-pass-3 this is
  transition-based replay; pass 3 will extend the ``sample()`` contract
  to R2D2-style burn-in sequences).

Ported from ``~/code/mkw/Wii-RL/BTR.py:311-733``. Algorithm preserved
line-by-line; only formatting, type hints, 4-space indent (the original
mixes 2 and 4), and method-docstring polish have been touched. The
commented-out priority_min code and alternate vectorized n-step helper
are preserved as comments so future diffs against VIPTankz stay legible.

Known deviations from the PER paper (Schaul et al. 2016)
---------------------------------------------------------
These are faithful to VIPTankz's published code and produce the BTR
paper's results; they are **not** faithful to the textbook PER spec.
Kept for now so our numbers are directly comparable to theirs. See
the 2026-04-21 forensic audit of VIPTankz/Wii-RL for full analysis.

1. **Importance-sampling exponent uses ``alpha`` instead of ``beta``**
   (``sample()`` line in this file; VIPTankz's ``BTR.py:622``).
   VIPTankz's comment acknowledges this was an accident; they kept
   it because "it performs better". Net effect: IS correction is
   weaker than the PER paper prescribes — training sits closer to
   vanilla prioritized sampling. If we ever see instability, swap
   to ``** -self.beta`` and restore a beta schedule.
2. **No beta annealing.** VIPTankz wires a ``priority_weight_increase``
   but never consumes it in the sampler (since the sampler uses
   alpha — see #1). We don't implement annealing either.
3. **Batch-min IS-weight normalization** instead of buffer-min. The
   code normalizes ``weights / weights.max()``; the theoretically-
   correct normalization uses the min priority in the entire buffer.
   VIPTankz comments at ``BTR.py:456-459`` describe the tradeoff
   (faster; "makes effectively no difference").
4. **Raw |δ| used as priority** (not Huber-adjusted). Dopamine and
   ku2482 use the quantile-Huber loss value instead; VIPTankz uses
   raw TD magnitude. Preserved.
5. **Storage multiplier default bumped from 1.25 → 1.75** for MKWii
   (episodes are ~1000 frames vs Atari's ~20). VIPTankz's 1.25 is
   under their own back-of-envelope minimum for long-episode tasks;
   we err generous. Configurable via constructor.

.. note::
    This module is pass-1 scope. Pass 3 will add an R2D2-style
    ``sample_sequences(batch_size, burn_in, learn_window)`` method that
    returns (burn_in + learn_window)-length sequences for recurrent
    training. The existing ``sample()`` stays as-is so a non-recurrent
    agent could still use this buffer.
"""

from __future__ import annotations

import numpy as np
import torch


class SumTree:
    """Binary tree where parents hold the sum of their children's priorities.

    Supports O(log n) prefix-sum traversal: given a target ``v`` in
    ``[0, total_priority)``, ``find()`` returns the leaf whose cumulative
    priority span contains ``v``. Used for proportional priority sampling.
    """

    def __init__(self, size: int, procgen: bool = False) -> None:  # procgen kept for parity
        self.index = 0
        self.size = size
        self.full = False  # tracks whether we've wrapped around
        # Put all used node leaves on the last tree level.
        self.tree_start = 2 ** (size - 1).bit_length() - 1
        self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
        self.max = 1  # initial max value to return (1 = 1^ω)

    # Update node values from current tree (vectorized helper).
    def _update_nodes(self, indices: np.ndarray) -> None:
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

    # Propagate changes up tree given tree indices.
    def _propagate(self, indices: np.ndarray) -> None:
        parents = (indices - 1) // 2
        unique_parents = np.unique(parents)
        self._update_nodes(unique_parents)
        if parents[0] != 0:
            self._propagate(parents)

    # Propagate a single value up the tree (scalar fast-path).
    def _propagate_index(self, index: int) -> None:
        parent = (index - 1) // 2
        left, right = 2 * parent + 1, 2 * parent + 2
        self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
        if parent != 0:
            self._propagate_index(parent)

    def update(self, indices: np.ndarray, values: np.ndarray) -> None:
        self.sum_tree[indices] = values
        self._propagate(indices)
        current_max_value = np.max(values)
        self.max = max(current_max_value, self.max)

    def _update_index(self, index: int, value: float) -> None:
        self.sum_tree[index] = value
        self._propagate_index(index)
        self.max = max(value, self.max)

    def append(self, value: float) -> None:
        self._update_index(self.index + self.tree_start, value)
        self.index = (self.index + 1) % self.size
        self.full = self.full or self.index == 0
        self.max = max(value, self.max)

    def _retrieve(self, indices: np.ndarray, values: np.ndarray) -> np.ndarray:
        children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
        # If indices correspond to leaf nodes, return them.
        if children_indices[0, 0] >= self.sum_tree.shape[0]:
            return indices
        # If children correspond to leaf nodes, bound rare outliers where the
        # total slightly overshoots due to float accumulation.
        if children_indices[0, 0] >= self.tree_start:
            children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
        left_children_values = self.sum_tree[children_indices[0]]
        successor_choices = np.greater(values, left_children_values).astype(np.int32)
        successor_indices = children_indices[successor_choices, np.arange(indices.size)]
        successor_values = values - successor_choices * left_children_values
        return self._retrieve(successor_indices, successor_values)

    def find(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (priorities, data_indices, tree_indices) for the leaves whose
        cumulative priority spans contain ``values``.
        """
        indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
        data_index = indices - self.tree_start
        return (self.sum_tree[indices], data_index, indices)

    def total(self) -> float:
        return self.sum_tree[0]


class PER:
    """Prioritized experience replay with frame-stacking and n-step returns.

    Designed for multi-env vectorized rollouts. Each ``stream`` ID carries
    its own pending window; the buffer stores individual frames in
    ``state_mem`` and writes transition pointers into ``pointer_mem``.

    This is VIPTankz's v1 transition-based implementation. Pass 3 extends
    with ``sample_sequences()`` for R2D2 recurrent replay; the transition-
    based ``sample()`` below stays as the fallback/reference path.
    """

    def __init__(
        self,
        size: int,
        device: str | torch.device,
        n: int,
        envs: int,
        gamma: float,
        alpha: float = 0.2,
        beta: float = 0.4,
        framestack: int = 4,
        imagex: int = 84,
        imagey: int = 84,
        rgb: bool = False,
        storage_size_multiplier: float = 1.75,  # VIPTankz default 1.25; bumped for MKWii
    ) -> None:
        self.st = SumTree(size)
        self.data = [None for _ in range(size)]
        self.index = 0
        self.size = size

        # Frame-pool storage is sized larger than the pointer table because
        # each stack spans 4 frames and n-step reaches further. VIPTankz's
        # original heuristic was size * 1.25 (discrete) or size * 4 (RGB).
        #
        # Comment from VIPTankz (preserved):
        # > the technical size to ensure there are no errors with overwritten
        # > memory in theory is very high — (2*framestack - overlap) *
        # > first_states + non_first_states. With N=3, framestack=4, size=1M,
        # > avg ep 20 → ~1.35M frame slots.
        #
        # MKWii episodes are ~1000+ frames (vs Atari's ~20), so frame pointers
        # can be written into faster than they're sampled for training,
        # potentially scrambling stored frame-stacks on the first buffer wrap.
        # We default to 1.75 (configurable) to give headroom. See the 2026-04-21
        # forensic audit for details.
        if rgb:
            self.storage_size = int(size * max(4.0, storage_size_multiplier * 4))
        else:
            self.storage_size = int(size * storage_size_multiplier)
        self.gamma = gamma
        self.capacity = 0

        self.point_mem_idx = 0
        self.state_mem_idx = 0
        self.reward_mem_idx = 0

        self.imagex = imagex
        self.imagey = imagey

        self.max_prio = 1

        self.framestack = framestack
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6  # small constant to stop 0 probability
        self.device = device

        self.last_terminal = [True for _ in range(envs)]
        self.tstep_counter = [0 for _ in range(envs)]

        self.n_step = n
        self.state_buffer: list[list[int]] = [[] for _ in range(envs)]
        self.reward_buffer: list[list[int]] = [[] for _ in range(envs)]

        if rgb:
            self.state_mem = np.zeros(
                (self.storage_size, 3, self.imagey, self.imagex), dtype=np.uint8
            )
        else:
            self.state_mem = np.zeros((self.storage_size, self.imagey, self.imagex), dtype=np.uint8)
        self.action_mem = np.zeros(self.storage_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.storage_size, dtype=float)
        self.done_mem = np.zeros(self.storage_size, dtype=bool)
        self.trun_mem = np.zeros(self.storage_size, dtype=bool)

        # pointer_mem: each entry holds indices into state_mem for the stack
        # and n_stack, plus indices into reward_mem for the n-step reward window.
        self.trans_dtype = np.dtype(
            [
                ("state", int, self.framestack),
                ("n_state", int, self.framestack),
                ("reward", int, self.n_step),
            ]
        )
        self.blank_trans = (
            np.zeros(self.framestack, dtype=int),
            np.zeros(self.framestack, dtype=int),
            np.zeros(self.n_step, dtype=int),
        )
        self.pointer_mem = np.array([self.blank_trans] * size, dtype=self.trans_dtype)

        self.overlap = self.framestack - self.n_step

        # The "technically correct" approach uses the min priority in the whole
        # buffer for weight normalization; VIPTankz uses the min from each batch
        # instead, which is much faster and empirically equivalent. Keeping the
        # original commented code as a reference.
        # self.priority_min = [float('inf') for _ in range(2 * self.size)]

    # -------- append path --------

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        n_state: np.ndarray,
        done: bool,
        trun: bool,
        stream: int,
        prio: bool = True,
    ) -> None:
        self.append_memory(state, action, reward, n_state, done, trun, stream)
        self.append_pointer(stream, prio)

        if done or trun:
            self.finalize_experiences(stream)
            self.state_buffer[stream] = []
            self.reward_buffer[stream] = []

        self.last_terminal[stream] = done or trun

    def append_pointer(self, stream: int, prio: bool) -> None:
        while (
            len(self.state_buffer[stream]) >= self.framestack + self.n_step
            and len(self.reward_buffer[stream]) >= self.n_step
        ):
            state_array = self.state_buffer[stream][: self.framestack]
            n_state_array = self.state_buffer[stream][self.n_step : self.n_step + self.framestack]
            reward_array = self.reward_buffer[stream][: self.n_step]

            self.pointer_mem[self.point_mem_idx] = (
                np.array(state_array, dtype=int),
                np.array(n_state_array, dtype=int),
                np.array(reward_array, dtype=int),
            )
            # self._set_priority_min(self.point_mem_idx, sqrt(self.max_prio))
            self.st.append(self.max_prio**self.alpha)

            self.capacity = min(self.size, self.capacity + 1)
            self.point_mem_idx = (self.point_mem_idx + 1) % self.size

            # Slide the window by one timestep.
            self.state_buffer[stream].pop(0)
            self.reward_buffer[stream].pop(0)
            # NOTE: VIPTankz's BTR.py:508 has `self.beta = 0` here. It's dead
            # code (beta is never read by sample() — see module-level deviation
            # #1). Removed for clarity. If we ever switch to `** -self.beta`
            # in the IS-weight formula and restore a proper beta schedule,
            # this location is wrong for the clobber anyway — the schedule
            # lives in the Agent, not the replay buffer.

    def finalize_experiences(self, stream: int) -> None:
        """Emit remaining experiences at episode boundary with zero-padded reward tail."""
        while (
            len(self.state_buffer[stream]) >= self.framestack
            and len(self.reward_buffer[stream]) > 0
        ):
            first_array = self.state_buffer[stream][: self.framestack]
            second_array = self.state_buffer[stream][-self.framestack :]
            reward_array = self.reward_buffer[stream][:]
            while len(reward_array) < self.n_step:
                reward_array.extend([0])

            self.pointer_mem[self.point_mem_idx] = (
                np.array(first_array, dtype=int),
                np.array(second_array, dtype=int),
                np.array(reward_array, dtype=int),
            )
            self.st.append(self.max_prio**self.alpha)

            self.point_mem_idx = (self.point_mem_idx + 1) % self.size
            self.capacity = min(self.size, self.capacity + 1)

            self.state_buffer[stream].pop(0)
            if len(self.reward_buffer[stream]) > 0:
                self.reward_buffer[stream].pop(0)

    def append_memory(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        n_state: np.ndarray,
        done: bool,
        trun: bool,
        stream: int,
    ) -> None:
        if self.last_terminal[stream]:
            # Start a fresh episode — write the whole frame stack + next frame.
            for i in range(self.framestack):
                self.state_mem[self.state_mem_idx] = state[i]
                self.state_buffer[stream].append(self.state_mem_idx)
                self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            # n_step not applied in this memory — just store the latest frame.
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

            self.tstep_counter[stream] = 0
        else:
            # Continuing episode — only the new next-frame + reward tuple.
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

    # -------- sample path --------

    def sample(self, batch_size: int, count: int = 0):
        """Sample ``batch_size`` transitions with prioritized stratified sampling.

        Returns (tree_idxs, states, actions, rewards, n_states, dones, weights)
        where states/n_states are on ``self.device`` as float32 uint8-cast-to-
        float, and rewards/dones/actions are device tensors of appropriate dtype.
        """
        p_total = self.st.total()

        # Stratified sampling over the priority distribution.
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length
        samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts

        prios, idxs, tree_idxs = self.st.find(samples)
        probs = prios / p_total

        # Dereference the pointer table.
        pointers = self.pointer_mem[idxs]
        state_pointers = np.array([p[0] for p in pointers])
        n_state_pointers = np.array([p[1] for p in pointers])
        reward_pointers = np.array([p[2] for p in pointers])
        if self.n_step > 1:
            action_pointers = np.array([p[2][0] for p in pointers])
        else:
            action_pointers = np.array([p[2] for p in pointers])

        states = torch.tensor(self.state_mem[state_pointers], dtype=torch.uint8)
        n_states = torch.tensor(self.state_mem[n_state_pointers], dtype=torch.uint8)

        rewards = self.reward_mem[reward_pointers]
        dones = self.done_mem[reward_pointers]
        truns = self.trun_mem[reward_pointers]
        actions = self.action_mem[action_pointers]

        if self.n_step > 1:
            rewards, dones = self.compute_discounted_rewards_batch(rewards, dones, truns)

        # Importance-sampling weights. VIPTankz notes that using alpha here
        # rather than beta was an accident that empirically performs better.
        # Kept verbatim; parameter semantics are VIPTankz-canonical.
        weights = (self.capacity * probs) ** -self.alpha
        weights = torch.tensor(
            weights / weights.max(),
            dtype=torch.float32,
            device=self.device,
        )

        if torch.isnan(weights).any():
            # Rare: sampled outside the filled range before the buffer was full.
            # Retry a couple of times before failing loudly.
            if count >= 5:
                raise RuntimeError("PER sample() produced NaN weights after retries")
            return self.sample(batch_size, count + 1)

        states = states.to(torch.float32).to(self.device)
        n_states = n_states.to(torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)

        return tree_idxs, states, actions, rewards, n_states, dones, weights

    def compute_discounted_rewards_batch(
        self,
        rewards_batch: np.ndarray,
        dones_batch: np.ndarray,
        truns_batch: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorizable-in-principle n-step discounting; VIPTankz uses the scalar loop for simplicity.

        A fully-vectorized alternative is preserved in the VIPTankz commit
        history (see ``compute_discounted_rewards_batch_no_loop`` in their
        BTR.py) but empirically it doesn't move the needle for batch 256 at
        n=3 — the Python loop is dominated by the torch.tensor constructions
        around it. If profiling says otherwise, swap in the vectorized path.
        """
        batch_size, n_step = rewards_batch.shape
        discounted_rewards = np.zeros(batch_size)
        cumulative_dones = np.zeros(batch_size, dtype=bool)

        for i in range(batch_size):
            cumulative_discount = 1
            for j in range(n_step):
                discounted_rewards[i] += cumulative_discount * rewards_batch[i, j]
                if dones_batch[i, j] == 1:
                    cumulative_dones[i] = True
                    break
                if truns_batch[i, j] == 1:
                    break
                cumulative_discount *= self.gamma

        return discounted_rewards, cumulative_dones

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray) -> None:
        priorities = priorities + self.eps
        # self._set_priority_min(idx - self.size + 1, sqrt(priority)) — see note above

        if np.isnan(priorities).any():
            # Surface as a loud log + keep going rather than crashing; VIPTankz's
            # training loops have logged this and recovered.
            import logging  # noqa: PLC0415 — local import, rarely hit

            logging.getLogger(__name__).warning(
                "NaN found in priorities; this usually means the loss diverged. "
                "priorities=%s", priorities
            )

        self.max_prio = max(self.max_prio, np.max(priorities))
        self.st.update(idxs, priorities**self.alpha)
