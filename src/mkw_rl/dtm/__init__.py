"""`.dtm` parsing + demo pipeline — **DORMANT** post-pivot.

Marked dormant on 2026-04-17 when the project moved to a multi-track
BTR-first plan; see ``docs/PIVOT_2026-04-17.md``. The code here is
complete and tested but is not on the active critical path — current
env + training code lives in ``mkw_rl.env`` and ``mkw_rl.rl``.

.. important::
    **Do not delete.** ``mkw_rl.bc.model`` imports ``N_STEERING_BINS``
    from ``mkw_rl.dtm.action_encoding``, and ``mkw_rl.rl.model``
    imports ``ImpalaEncoder`` from ``mkw_rl.bc.model`` (transitively
    bringing in this module). The chain is load-bearing for the
    active BTR path even while the full BC pipeline sits dormant.

Revival path: for BC augmentation during BTR training or TAS-demo
loading (see ``docs/PIVOT_2026-04-17.md`` "Future BC path"), this
module provides .dtm parsing, frame loading, demo pairing, sequence
dataset, action encoding, and overlay visualization.
"""
