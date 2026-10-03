# Claude Execution Rules — Phase 4

1. Read Phase 0–3 docs before modifying code.
2. Treat Phase 3 commit `fdfce67` and its harness/protocol as baseline.
3. Reuse the Phase 3 framework; do not rewrite it without a demonstrated defect.
4. Do not integrate Feature-Match.
5. Do not select a production model early.
6. B0 untouched input is always a first-class baseline.
7. Identity utility is primary; perceptual quality is secondary.
8. Keep synthetic and real-world populations separate.
9. Freeze manifests before final evaluation.
10. Never tune thresholds on final test data.
11. Record model, weight, code, config and dataset hashes.
12. New models require licence screening before benchmark use.
13. External datasets require legal/permission evidence.
14. Do not claim fairness from insufficient sample sizes.
15. Preserve reproducibility and deterministic re-runs.
16. Each task must produce a report and machine-readable evidence where appropriate.
17. If blocked, stop at the boundary and document the blocker; do not silently substitute an unapproved source/model/dataset.
18. Keep commits scoped to the task.
19. At task end report: files changed, tests, evidence, blockers, gate status, next task.
20. Do not push to a remote unless one is explicitly configured and authorized.
