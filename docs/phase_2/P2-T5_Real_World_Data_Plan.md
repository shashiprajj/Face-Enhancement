# P2-T5 — Real-World Low-Quality Face Test Set

| Field | Value |
|---|---|
| Phase / Task | Phase 2 / P2-T5 |
| Status | Complete as a **plan and executable specification**. **No real-world data acquired** (none is approved). |
| Date | 2026-09-28 |
| Executable spec | `configs/phase2/realworld_collection_v1.yaml` (`rw-spec-1.0.0`) |
| Validator | `scripts/p2_validate_collection.py` (unit-tested in P2-T6's test suite) |
| Templates | `docs/phase_2/templates/` (consent DRAFT, participants/captures CSV) |

---

## 1. The Gap

The Phase 0 master specification (§8) makes **real-world degraded evaluation mandatory**. Phase 1 closed with gap **G-3**: there is no eligible real-world dataset. The Phase 2 README states that *synthetic degradation must not be treated as a substitute*. P2-T4's synthetic set therefore does not close this gap.

## 2. Source / License Decision Matrix

Re-checked 2026-09-28 against primary sources where reachable.

| Option | Real LQ? | Same-identity HQ reference? | License / terms evidence | Intended use fit | Status | Approval |
|---|---|---|---|---|---|---|
| **DS03 Internal consent-based collection** | ✅ | ✅ (captured in-session) | Company-owned under explicit consent scope | ✅ evaluation-only scope | **ELIGIBLE once consent program is approved** | Needs Legal/Compliance approval of the consent form + DPIA-style review |
| DS12 **SCface** | ✅ surveillance cameras | ✅ mugshots | Release agreement (PDF text extracted today): *"The SCface database, in whole or in part, will not be used for any commercial purpose in any form."* Also bars redistribution even within the requesting organization. | ❌ | **INELIGIBLE** (upgraded from REVIEW_REQUIRED in P1-T6) | — |
| DS05 **Casual Conversations v2** (Meta) | ✅ varied devices/lighting | ✅ multiple videos per participant | Official page: *"Limited; see full license language"*. A secondary source describes it as for *"non-commercial, academic research"*. The full agreement is behind Meta's download portal (not reviewed). | ⚠️ likely not | **REVIEW_REQUIRED (leaning INELIGIBLE)** | Reviewer must read the full agreement |
| DS11 TinyFace | ✅ native LR | ❌ no HR GT | No terms stated | Partial | REVIEW_REQUIRED | — |
| DS13 QMUL-SurvFace | ✅ | ❌ | No terms seen | Partial | REVIEW_REQUIRED | — |
| DS14 WIDER FACE / WIDER-Test | ✅ | ❌ | CC BY-NC-ND 4.0 | ❌ | INELIGIBLE | — |
| DS09/10 LFW / XQLFW | mild | pairs | No license grant / "All Rights Reserved" | ⚠️ | REVIEW_REQUIRED | — |
| Commercial data vendor (licensed, consented face capture) | ✅ | ✅ (spec-able) | Contractual | ✅ | Option (not evaluated) | Procurement + Legal |

**Conclusion: no external source is currently approvable for commercial evaluation.** Per the task instruction, instead of fabricating data this task delivers a concrete, executable **internal collection specification**, plus the option of procuring an equivalent vendor collection built to the same spec.

`dataset_license_matrix.csv` (Phase 1) remains the historical record. The SCface change is recorded here and in the Phase 2 final report.

## 3. Internal Collection Specification (summary of `rw-spec-1.0.0`)

**Purpose:** evaluation only. Training requires a separate optional consent scope.

**Capture protocol per participant**

| Role | Conditions (code: levels) | Devices |
|---|---|---|
| HQ_REF (clean target) | 1–1.5 m, ≥ 400 lux, frontal + yaw ±30°, 2 shots per pose. **Validator enforces one face with inter-ocular distance ≥ 120 px.** | mirrorless/DSLR or flagship phone rear |
| LQ | DIST: 3 / 6 / 10 m | flagship + budget phone rear |
| LQ | LOWL: 50 / 10 lux | budget phone front, 720p webcam |
| LQ | MOTN: subject turn, handheld 1/15 s | budget phone |
| LQ | DFOC: focus offset | mirrorless |
| LQ | COMP: messaging app ×1, ×3; 480p webcam stream | any |
| LQ | LCDV: 480p webcam, IP camera at 2.5 m height | webcam, IP camera |
| LQ | BKLT: window backlight | phone |
| LQ (optional) | OCCL: glasses, mask, hand (P0-T3 E5 edge case) | phone |

Every LQ capture links to an HQ_REF of the **same participant** (validator-enforced). Each participant must cover DIST, LOWL, MOTN, COMP, and LCDV (checked by the validator).

**Consent and documentation**
- Signed forms are stored in a legal-restricted store, **never** in the dataset.
- The dataset carries only `consent_ref_id` and `consent_scopes`. The validator enforces the `evaluation` scope and a `withdrawn` flag; withdrawn participants are excluded from the manifest.
- Minimum age 18.

**Anonymized IDs:** `P-xxxxxx` (participant), `S-xxxxxx` (session), `C-xxxxxxxx` (capture), `D-…` (device). All are random and never derived from identity. The validator enforces the formats.

**Data minimization** (validator-enforced):
- Forbidden columns: name, email, phone, address, DOB, Aadhaar/PAN/passport, and similar.
- Every cell is regex-scanned for e-mail / Indian mobile numbers.
- Only **age bands** are allowed, never exact ages.
- Images carrying **GPS EXIF are rejected**.
- Optional demographics are self-reported, including the Monk Skin Tone scale, and used for aggregate fairness only.

**Storage and retention**
- Encrypted at rest, internal only, named team with access logging.
- Not uploaded to rented third-party GPUs without approval.
- Deleted at project end + 90 days, or within 30 days of withdrawal.
- No sample images in reports without the `publication_of_samples` scope.

**Splits:** by participant, `sha256(salt + participant_id) mod 100 < 30 → dev`, else test. No training split. No participant can appear in two splits.

**Sample size:** 30-participant pilot (a justified estimate for variance estimation). The final N is **TBD** by power analysis (P0-T4 TH-11). No final number is invented.

## 4. How the Validator Makes This Executable

```
scripts/dock.sh python scripts/p2_validate_collection.py --root <collection> --manifest data/manifests/ds03_v1.csv.gz
```

It checks structure, IDs, consent scope, PII columns and values, GPS EXIF, HQ face/resolution, same-participant pairing, condition coverage, and file readability. It assigns participant-level splits and writes an **acquisition manifest** (SHA-256 per file, spec version), but only when there are zero hard errors. Unit tests with synthetic fixtures cover passing and failing collections (P2-T6 test suite). The fixtures are generated test images only, **not data**.

## 5. Legally Reviewable Path — Next Steps

1. Legal/Compliance review of `consent_form_DRAFT.md` and this spec (DPDP Act 2023: notice, purpose limitation, retention, withdrawal).
2. Approve the storage location and access list.
3. Run the pilot (30 participants), then `p2_validate_collection.py`.
4. Use pilot variance to set the final N, and to **calibrate the synthetic degradation severities** (P2-T4 §6).
5. In parallel: the reviewer reads the Casual Conversations v2 agreement; Procurement may quote a vendor collection built to `rw-spec-1.0.0`.

## 6. Done-When Check

There is a documented, legally reviewable path to real-world low-quality evaluation data: a decision matrix, an executable spec with an enforcing validator, a draft consent, and templates. No data was fabricated. The gap remains **open** until collection is approved and executed. ✅ (path) / ❌ (data)
