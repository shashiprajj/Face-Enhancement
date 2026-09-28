PHASE 1 — FACE ENHANCEMENT SERVICE
MODEL & DATASET DISCOVERY + ENVIRONMENT

EXECUTION ORDER
1. P1-T1 Phase 0 Open Items & Project Baseline
2. P1-T2 Hugging Face / Open-Source Model Survey
3. P1-T3 Model License & Commercial-Use Screening
4. P1-T4 Face Detection / Alignment Survey
5. P1-T5 Restoration Model Shortlist & Test Design
6. P1-T6 Dataset Survey & License Screening
7. P1-T7 Dataset Selection & Evaluation Plan
8. P1-T8 Local ML Environment & Infrastructure Setup
9. P1-T9 Approved Model & Dataset Acquisition
10. P1-T10 Phase 1 Final Inventory & Readiness Report

CORE RULES
- Phase 0 Master Specification is authoritative.
- Feature-Match is completely outside this project.
- Do not implement production FastAPI in Phase 1.
- Do not choose a final model by intuition.
- License screening is a hard gate.
- Never download unresolved-license assets.
- Never put Hugging Face/API tokens in source code or Git.
- Separate technical suitability from legal/commercial eligibility.
- Phase 1 ends with a benchmark-ready candidate set, not a final production model.

HUGGING FACE ACCESS
If authentication is needed, use an explicit supported local authentication mechanism. A browser login does not automatically grant command-line access. Do not bypass authentication or security controls.

EXPECTED FINAL OUTPUT
reports/Phase_1_Final_Report.md

The final report must state GREEN / YELLOW / RED readiness for Phase 2.
