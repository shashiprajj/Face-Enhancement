FACE ENHANCEMENT SERVICE — PHASE 5 RAPID VALIDATION

Purpose
-------
This is the compressed Phase 5 plan designed for the current time constraint.

The goal is NOT full production certification yet.
The goal is to answer one decisive question quickly:

Can the existing A4 + T6 + T7 pipeline take genuinely poor-quality face images, enhance only the appropriate ones, and improve the actual Feature-Match result?

Current baseline
----------------
- Phase 4 is complete and YELLOW.
- T6 Do-Not-Enhance gate is frozen.
- T7 selective routing is frozen.
- A4 is the only current enhancement candidate.
- A1/A2/A5 must not be routed.
- CPU-only execution is acceptable.
- Licensing remains a production-approval track and does not block local technical experimentation.
- The large synthetic benchmark work has already been completed and must NOT be repeated.

Phase 5 Rapid Validation contains 4 tasks:
R5-1 Standalone Working Enhancement Pipeline
R5-2 Small Real-World Feature-Match Experiment
R5-3 Targeted Fix + Robustness Pass
R5-4 Final Rapid Validation Gate

Target effort
-------------
Approximately 7–11 hours of engineering work, plus CPU inference time that can run unattended.

Execution rule
--------------
Execute tasks sequentially.
For each task:
1. Inspect the existing implementation first.
2. Make only the changes required by the task.
3. Run relevant tests/validation.
4. Produce a concise report.
5. Commit the intended changes.
6. Stop at the task acceptance gate.

Do not expand this into another large model-discovery or benchmark program.
