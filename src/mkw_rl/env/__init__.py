"""Multi-track BTR environment: gym-compliant wrapper around VIPTankz's Dolphin scripting fork.

See ``docs/TRAINING_METHODOLOGY.md`` for the algorithmic choices (variable
checkpoints, lenient reset, reward structure) and ``docs/PIVOT_2026-04-17.md``
for why we own this code rather than importing VIPTankz's verbatim.

Components:
- ``track_meta`` — loads ``data/track_metadata.yaml`` (WR times → checkpoint counts)
- ``reward`` — per-episode reward accumulator implementing the v2 formula
- ``dolphin_script`` — runs inside Dolphin via ``--script`` (the slave process)
- ``dolphin_env`` — gymnasium-compatible env launched in the training process (master)

.. important::
    This ``__init__.py`` does NOT auto-import the heavy master-side modules
    (``dolphin_env`` pulls in gymnasium + subprocess + socket). The slave
    Python (inside Dolphin via ``--script``) imports ``mkw_rl.env.reward`` /
    ``mkw_rl.env.track_meta`` directly — which triggers THIS ``__init__.py``.
    Auto-importing ``dolphin_env`` here caused the slave to hang on loading
    the gym machinery (which it never needs and which trips on Dolphin's
    embedded Python's ABI for gymnasium's ctypes-heavy init path). Callers
    needing the master-side env class should do
    ``from mkw_rl.env.dolphin_env import MkwDolphinEnv`` explicitly.
"""
