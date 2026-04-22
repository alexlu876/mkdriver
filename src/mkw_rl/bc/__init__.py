"""Behavioral cloning module — **DORMANT** post-pivot.

Marked dormant on 2026-04-17 when the project moved to a multi-track
BTR-first plan; see ``docs/PIVOT_2026-04-17.md``. The code here is
complete and tested (124+ tests pass) but is not on the active
critical path — the training loop and scripts target ``mkw_rl.rl``
instead.

.. important::
    **Do not delete.** ``mkw_rl.rl.model.BTRPolicy`` imports
    ``ImpalaEncoder`` from ``mkw_rl.bc.model`` for BC↔BTR weight
    compatibility. Deleting ``bc.model`` breaks the active BTR
    forward pass immediately. See ``src/mkw_rl/rl/model.py:40``.

Revival path: if we ever want BC augmentation during BTR training
(see ``docs/PIVOT_2026-04-17.md`` "Future BC path"), this module
provides the model, dataset, and training loop as-is.
"""
