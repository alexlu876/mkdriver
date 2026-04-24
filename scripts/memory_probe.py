#!/usr/bin/env python3
"""Standalone BTRAgent.learn_step memory probe — no Dolphin needed.

Reproduces the OOM path in seconds instead of the 15-minute warmup
shakedown cycle. Fakes a replay buffer primed with random uint8 frames,
then runs learn_step(s) with torch.cuda.memory_allocated() printed at
every named checkpoint inside the step.

Usage
-----
::

    # On the 5090 box:
    LD_LIBRARY_PATH=.venv/lib/python3.13/site-packages/nvidia/cudnn/lib:\\
        /.uv/python_install/cpython-3.13-linux-x86_64-gnu/lib:$LD_LIBRARY_PATH \\
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
        .venv/bin/python scripts/memory_probe.py --batch-size 128

Output: a table of cumulative memory allocation after each phase of
learn_step, plus a final memory_summary dump showing the allocator's
view of where memory is sitting. Clearly identifies the hot spots
(input staging, encoder forward, LSTM unroll, backward) so we know
what to attack.
"""

from __future__ import annotations

import argparse
import contextlib
import sys

import numpy as np
import torch


def _mb(n_bytes: int) -> str:
    return f"{n_bytes / (1024**2):>8.1f} MiB"


def _print_mem(label: str, baseline: int = 0) -> int:
    now = torch.cuda.memory_allocated()
    peak = torch.cuda.max_memory_allocated()
    delta = now - baseline if baseline else 0
    print(
        f"{label:<45}  now={_mb(now)}  peak={_mb(peak)}"
        + (f"  +{_mb(delta)}" if baseline else "")
    )
    return now


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--burn-in", type=int, default=20)
    ap.add_argument("--learning-seq-len", type=int, default=40)
    ap.add_argument("--framestack", type=int, default=4)
    ap.add_argument("--imagey", type=int, default=75)
    ap.add_argument("--imagex", type=int, default=140)
    ap.add_argument("--feature-dim", type=int, default=256)
    ap.add_argument("--lstm-hidden", type=int, default=512)
    ap.add_argument("--linear-size", type=int, default=512)
    ap.add_argument(
        "--encoder-channels", type=str, default="32,64,64",
        help="Comma-separated encoder channel widths",
    )
    ap.add_argument("--num-tau", type=int, default=8)
    ap.add_argument("--n-cos", type=int, default=64)
    ap.add_argument("--replay-size", type=int, default=65536,
                    help="Small buffer — we prime just enough for sampling")
    ap.add_argument("--autocast", choices=["off", "bf16", "fp16"], default="bf16")
    ap.add_argument("--steps", type=int, default=2, help="How many learn_steps to run")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("error: needs CUDA for memory probing", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU total: {_mb(torch.cuda.get_device_properties(0).total_memory)}")
    print(f"torch: {torch.__version__}")
    print("=" * 80)

    from mkw_rl.rl.model import BTRConfig, BTRPolicy
    from mkw_rl.rl.replay import PER

    seq_len = args.burn_in + args.learning_seq_len
    enc_ch = tuple(int(x) for x in args.encoder_channels.split(","))

    _print_mem("00 baseline (after imports)")

    # Build online + target policies.
    cfg = BTRConfig(
        stack_size=args.framestack,
        input_hw=(args.imagey, args.imagex),
        encoder_channels=enc_ch,
        feature_dim=args.feature_dim,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=1,
        linear_size=args.linear_size,
        num_tau=args.num_tau,
        n_cos=args.n_cos,
        n_actions=40,
        layer_norm=True,
        spectral_norm=True,
    )
    online = BTRPolicy(cfg).to(device)
    target = BTRPolicy(cfg).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    for m in target.modules():
        # disable noise on target the way the real agent does
        if hasattr(m, "weight_epsilon"):
            m.weight_epsilon.zero_()
            m.bias_epsilon.zero_()
    _print_mem("01 after online + target nets on GPU")

    optimizer = torch.optim.Adam(online.parameters(), lr=1e-4, eps=0.005 / args.batch_size)
    # touch a gradient so optimizer state is actually allocated
    dummy = torch.zeros(1, 1, args.framestack, args.imagey, args.imagex,
                        dtype=torch.uint8, device=device)
    with torch.no_grad():
        pass  # keep optimizer state unallocated until the real backward below
    _print_mem("02 after optimizer construction")

    # Build a small PER and prime it with random data.
    # Use bigger storage multiplier so the pointer table / sum tree / frame pool
    # for a warmup-filled buffer isn't part of the measurement.
    replay = PER(
        size=args.replay_size,
        device=device,
        n=3,
        envs=4,
        gamma=0.997,
        alpha=0.2,
        beta=0.4,
        framestack=args.framestack,
        imagex=args.imagex,
        imagey=args.imagey,
        storage_size_multiplier=1.75,
    )
    _print_mem("03 after PER construction (CPU-side buffers)")

    rng = np.random.default_rng(0)
    # Fill replay up to >> seq_len so sample_sequences works.
    n_prime = max(seq_len * 8, 4096)
    print(f"priming replay with {n_prime} transitions per stream (4 streams)...")
    for stream in range(4):
        for i in range(n_prime):
            state = rng.integers(0, 256, size=(args.framestack, args.imagey, args.imagex), dtype=np.uint8)
            n_state = rng.integers(0, 256, size=(args.framestack, args.imagey, args.imagex), dtype=np.uint8)
            replay.append(
                state=state, action=i % 40, reward=float(i % 10) * 0.1,
                n_state=n_state, done=(i == n_prime - 1), trun=False, stream=stream,
            )
    _print_mem("04 after priming replay (should still be ~same; CPU-side)")

    # Set up autocast.
    if args.autocast == "bf16":
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # noqa: E731
    elif args.autocast == "fp16":
        amp_ctx = lambda: torch.autocast(device_type="cuda", dtype=torch.float16)  # noqa: E731
    else:
        amp_ctx = lambda: contextlib.nullcontext()  # noqa: E731
    print(f"\nAutocast mode: {args.autocast}\n")

    # Run learn_step-like phases manually with instrumentation.
    for step in range(args.steps):
        print(f"\n=== learn_step {step + 1}/{args.steps} ===")
        base = _print_mem(f"  {step}.0 entry")

        # Sample sequences (allocates GPU tensors for the batch).
        _, states, actions, rewards, n_states, dones, weights = replay.sample_sequences(
            args.batch_size, seq_len,
        )
        _print_mem(f"  {step}.1 after sample_sequences", base)

        burn_in = args.burn_in
        burn_states = states[:, :burn_in]
        burn_n_states = n_states[:, :burn_in]
        learn_states = states[:, burn_in:]
        learn_n_states = n_states[:, burn_in:]

        online.reset_noise()
        _print_mem(f"  {step}.2 after reset_noise", base)

        with amp_ctx():
            with torch.no_grad():
                _, _, hidden_online = online(burn_states)
                _, _, hidden_target = target(burn_n_states)
            _print_mem(f"  {step}.3 after burn_in fwd (no_grad)", base)

            online_q, online_tau, _ = online(learn_states, hidden=hidden_online)
            _print_mem(f"  {step}.4 after online learn fwd (grad)", base)

            with torch.no_grad():
                target_q, _, _ = target(learn_n_states, hidden=hidden_target)
            _print_mem(f"  {step}.5 after target learn fwd (no_grad)", base)

            loss = (online_q.mean() - target_q.mean()) ** 2  # dummy but exercises grad
            _print_mem(f"  {step}.6 after dummy loss", base)

        if args.autocast != "off":
            loss = loss.float()

        optimizer.zero_grad()
        loss.backward()
        _print_mem(f"  {step}.7 after backward", base)
        optimizer.step()
        _print_mem(f"  {step}.8 after optimizer.step", base)

        # Free explicit references so step N+1 starts clean.
        del states, actions, rewards, n_states, dones, weights
        del burn_states, burn_n_states, learn_states, learn_n_states
        del hidden_online, hidden_target, online_q, online_tau, target_q, loss
        torch.cuda.synchronize()
        _print_mem(f"  {step}.9 after del + sync", base)

    print("\n" + "=" * 80)
    print("FINAL torch.cuda.memory_summary():")
    print(torch.cuda.memory_summary(abbreviated=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
