# Vendored KDFlow

`KDFlow/` is vendored from https://github.com/songmzhang/KDFlow
at upstream commit `037e18f5bb1ae432e4dcfcea9fbe277e0e907703` (cloned 2026-07-03/04, MIT license).

It is vendored (not a submodule) because our Phase-3 (per-teacher weighting baseline) and
Phase-4 (direction/magnitude-decoupled aggregation) modifications live inside its on-policy
trainer. The gradient combine point is mapped in `NOTES_kdflow_combine_point.md`
(anchor: `KDFlow/kdflow/ray/train/student_actor.py:286-327`).

To diff against upstream later: `git clone https://github.com/songmzhang/KDFlow` and compare.
