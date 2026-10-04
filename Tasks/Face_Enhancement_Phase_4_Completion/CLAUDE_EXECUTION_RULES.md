# Claude Execution Rules

1. Read existing Phase 0–4 documentation first.
2. Do not redesign completed work without evidence.
3. Do not start Phase 5 during T6–T8.
4. Resume partial inference; do not recompute completed pairs.
5. Use development data for calibration only.
6. Freeze T6 before evaluation.
7. Never tune thresholds from evaluation/test results.
8. Keep B0/original as a first-class baseline.
9. Do not rank models from a single metric.
10. Report SFace and dlib separately.
11. Missing evidence is unresolved, not a pass.
12. Add tests for every deterministic rule.
13. Keep changes scoped to the task.
14. Commit each task separately.
15. At completion report files changed, tests, commands, artifacts, evidence status and commit hash.
16. Under memory pressure reduce workers and resume safely; do not silently change the protocol.
